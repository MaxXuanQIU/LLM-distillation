import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from kd.common import (collate, config, generate, load_model, provenance, read_jsonl,
                       seed_all, tokenizer, training_parts, write_json, write_jsonl)
from kd.loss import response_loss


@torch.no_grad()
def validation(model, tok, rows, cfg):
    model.eval()
    total, tokens = 0., 0
    for start in range(0, len(rows), cfg["micro_batch_size"]):
        batch = rows[start:start+cfg["micro_batch_size"]]
        completions = [r["completion_ids"] for r in batch]
        for part, comp in training_parts(batch, completions, cfg.get("max_training_batch_tokens", float("inf"))):
            ids, mask, labels = collate(part, comp, tok.pad_token_id, model.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, n = response_loss(model, None, ids, mask, labels, cfg["loss_chunk_size"])
            total += loss.item()
            tokens += n
    model.train()
    return total / tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["E1", "E2", "E3", "E4"], required=True)
    p.add_argument("--pilot-steps", type=int, default=0)
    p.add_argument("--config", default="config.json")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = config(args.config)
    seed_all(cfg["seed"])
    out = Path("runs") / (("pilot_" if args.pilot_steps else "") + args.experiment)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise RuntimeError(f"Completed run already exists: {out}")
    student = load_model(cfg["student"], "cuda:0", train=True)
    teacher = None if args.experiment == "E1" else load_model(cfg["teacher"], "cuda:1")
    tok = tokenizer(cfg["student"])
    assert all(p.requires_grad for p in student.parameters())
    if teacher is not None:
        assert not any(p.requires_grad for p in teacher.parameters())
    train = read_jsonl("data/train.jsonl")
    val = read_jsonl("data/validation.jsonl")
    random.Random(cfg["seed"]).shuffle(train)
    effective = cfg["effective_batch_size"]
    micro = cfg["micro_batch_size"]
    assert effective % micro == 0
    steps = math.ceil(len(train)/effective)
    if args.pilot_steps:
        steps = args.pilot_steps
        train = train[:steps*effective]
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"], foreach=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, math.ceil(steps*cfg["warmup_ratio"])), steps)
    mix_rng = random.Random(cfg["seed"] + 1)
    lmbda = {"E1": 0., "E2": 0., "E3": .5, "E4": 1.}[args.experiment]
    history = []
    first_step, total_tokens, on_examples, generation_seconds = 0, 0, 0, 0.
    previous_seconds = 0.
    state_path = out / "resume.pt"
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_step, total_tokens, on_examples = state["step"], state["total_tokens"], state["on_examples"]
        generation_seconds, previous_seconds = state["generation_seconds"], state["seconds"]
        mix_rng.setstate(state["mix_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        history = state["history"]
        del state
    else:
        write_json(out/"metadata.json", {"config": cfg, "experiment": args.experiment,
                   "lambda": lmbda, "beta": 0, "trainable_parameters": sum(p.numel() for p in student.parameters()),
                   "precision": "FP32 master parameters and AdamW states; BF16 autocast activations", **provenance()})
    start = time.monotonic()
    trajectory_path = out / "trajectories.jsonl"
    prior = read_jsonl(trajectory_path) if first_step and trajectory_path.exists() else []
    write_jsonl(trajectory_path, [r for r in prior if r["step"] <= first_step])
    student.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(first_step, steps):
        batch = train[step*effective:(step+1)*effective]
        step_loss, step_tokens, step_on, step_generation = 0., 0, 0, 0.
        trajectories = []
        for offset in range(0, len(batch), micro):
            part = batch[offset:offset+micro]
            on_policy = mix_rng.random() < lmbda
            if on_policy:
                gen_start = time.monotonic()
                comp = generate(student, tok, part, cfg["max_completion_tokens"], cfg)
                step_generation += time.monotonic() - gen_start
                step_on += len(part)
            else:
                comp = [r["completion_ids"] for r in part]
            for row, completion in zip(part, comp):
                trajectories.append({"step": step+1, "id": row["id"], "on_policy": on_policy,
                    "completion_ids": completion, "truncated": completion[-1] != tok.eos_token_id})
            for piece, targets in training_parts(part, comp, cfg.get("max_training_batch_tokens", float("inf"))):
                ids, mask, labels = collate(piece, targets, tok.pad_token_id, student.device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, n = response_loss(student, teacher, ids, mask, labels,
                                            cfg["loss_chunk_size"], cfg["kd_temperature"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                loss.backward()
                step_loss += loss.item()
                step_tokens += n
                del loss, ids, mask, labels
        for param in student.parameters():
            if param.grad is not None:
                param.grad.div_(step_tokens)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_tokens += step_tokens
        on_examples += step_on
        generation_seconds += step_generation
        elapsed = previous_seconds + time.monotonic() - start
        record = {"step": step+1, "loss": step_loss/step_tokens, "lr": lr,
                  "grad_norm": float(grad_norm), "tokens": step_tokens, "total_tokens": total_tokens,
                  "on_policy_examples": step_on, "generation_seconds": step_generation,
                  "seconds": elapsed, "peak_memory_gb": [torch.cuda.max_memory_allocated(i)/1e9
                                                          for i in range(torch.cuda.device_count())]}
        history.append(record)
        with trajectory_path.open("a") as f:
            for trajectory in trajectories:
                f.write(json.dumps(trajectory) + "\n")
        write_json(out/"history.json", history)
        print(json.dumps(record), flush=True)
        if not args.pilot_steps and ((step+1) % 50 == 0 or step+1 == steps):
            state = {"model": student.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "step": step+1, "total_tokens": total_tokens,
                     "on_examples": on_examples, "generation_seconds": generation_seconds,
                     "seconds": elapsed, "mix_rng": mix_rng.getstate(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "history": history}
            torch.save(state, out/"resume.tmp")
            (out/"resume.tmp").replace(state_path)
    training_seconds = previous_seconds + time.monotonic()-start
    val_ce = validation(student, tok, val[:16] if args.pilot_steps else val, cfg)
    if not args.pilot_steps:
        student.save_pretrained(out/"checkpoint", safe_serialization=True)
        tok.save_pretrained(out/"checkpoint")
    write_json(out/"summary.json", {"experiment": args.experiment, "lambda": lmbda,
               "optimizer_steps": steps, "examples": len(train), "completion_tokens": total_tokens,
               "on_policy_examples": on_examples, "training_seconds": training_seconds,
               "generation_seconds": generation_seconds, "allocated_gpu_hours": training_seconds*
               torch.cuda.device_count()/3600, "validation_cross_entropy": val_ce,
               "peak_memory_gb": [torch.cuda.max_memory_allocated(i)/1e9 for i in range(torch.cuda.device_count())]})


if __name__ == "__main__":
    main()
