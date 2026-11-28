"""Matched full-parameter off-policy training and fixed-trajectory replay."""
import argparse
import hashlib
import math
import random
import time

import torch
from transformers import get_cosine_schedule_with_warmup

from kd.common import (collate, config, digest_file, load_model, provenance, read_jsonl,
                       seed_all, tokenizer, training_parts, write_json)
from kd.loss import response_loss
from kd.train import validation
from studies.followup.common import OUT, RAW, RATES, POINTS, load


def run(arm):
    cfg = config()
    seed_all(cfg["seed"])
    summary = OUT / "training" / f"{arm}.json"
    if summary.exists():
        return
    fixed = read_jsonl("data/train.jsonl")
    random.Random(cfg["seed"]).shuffle(fixed)
    valid = read_jsonl("data/validation.jsonl")
    batch_size = cfg["effective_batch_size"]
    steps = math.ceil(len(fixed) / batch_size)
    assert steps == 157 and batch_size == cfg["micro_batch_size"] == 32
    replay = arm.startswith("replay-")
    equal = arm == "replay-answer"
    points = (32,) if replay else POINTS
    if replay:
        trajectories = read_jsonl("runs/E4/trajectories.jsonl")[:32 * batch_size]
        assert len(trajectories) == 32 * batch_size
        for idx, row in enumerate(trajectories):
            assert row["step"] == idx // batch_size + 1 and row["on_policy"] and row["id"] == fixed[idx]["id"]
            fixed[idx] = {**fixed[idx], "completion_ids": row["completion_ids"], "truncated": row["truncated"]}
    student = load_model(cfg["student"], "cuda:0", train=True)
    teacher = load_model(cfg["teacher"], "cuda:1")
    tok = tokenizer(cfg["student"])
    reference = load("runs/E4/history.json" if replay else "runs/E2/history.json")
    lr = RATES.get(arm, cfg["learning_rate"])
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=cfg["weight_decay"], foreach=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(steps*cfg["warmup_ratio"]), steps)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    history, checkpoints = [], {}
    start = time.monotonic()
    for step in range(1, max(points) + 1):
        batch = fixed[(step-1)*batch_size:step*batch_size]
        comps = [r["completion_ids"] for r in batch]
        tokens, summed = 0, 0.
        for piece, targets in training_parts(batch, comps, cfg["max_training_batch_tokens"]):
            ids, mask, labels = collate(piece, targets, tok.pad_token_id, student.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, count = response_loss(student, teacher, ids, mask, labels, cfg["loss_chunk_size"],
                                            cfg["kd_temperature"], equal_answers=equal)
            if not torch.isfinite(loss):
                raise FloatingPointError(step)
            loss.backward()
            tokens += count
            summed += loss.item()
            del loss, ids, mask, labels
        assert tokens == reference[step-1]["tokens"]
        denominator = len(batch) if equal else tokens
        for param in student.parameters():
            if param.grad is not None:
                param.grad.div_(denominator)
        norm = torch.nn.utils.clip_grad_norm_(student.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True)
        update_lr = scheduler.get_last_lr()[0]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(dict(step=step, tokens=tokens, capped=sum(r["truncated"] for r in batch),
                            answers=len(batch), loss=summed/denominator, gradient_norm=float(norm), lr=update_lr))
        print(arm, step, tokens, flush=True)
        if step in points:
            target = RAW / "checkpoints" / f"{arm}-{step}"
            target.mkdir(exist_ok=True, parents=True)
            student.save_pretrained(target, safe_serialization=True)
            tok.save_pretrained(target)
            ce = validation(student, tok, valid, cfg)
            checkpoints[str(step)] = {"validation_ce": ce, "sha256": digest_file(target / "model.safetensors")}
    rows = fixed[:max(points)*batch_size]
    write_json(summary, dict(arm=arm, peak_lr=lr, scheduler_steps=steps, initial_full_schedule=True,
                             equal_answer_weight=equal, seed=cfg["seed"], seconds_including_validation_and_export=time.monotonic()-start,
                             ordered_ids_sha256=hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(),
                             checkpoints=checkpoints, history=history, **provenance()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=[*RATES, "replay-answer", "replay-token"])
    run(p.parse_args().arm)
