import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lm_eval.tasks.ifeval.utils import process_results

from kd.common import (config, generate, load_model, prompt_ids, provenance,
                       read_jsonl, seed_all, tokenizer, write_json, write_jsonl)


@torch.no_grad()
def evaluate_arc(model, tok, rows):
    records = []
    for row in rows:
        letters = "ABCDE"[:len(row["choices"]["text"])]
        prompt = "Choose the correct answer to the following question. Respond with only the letter of the answer.\n\n"
        prompt += row["question"] + "\n"
        prompt += "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, row["choices"]["text"]))
        ids = torch.tensor([prompt_ids(tok, prompt)], device=model.device)
        logits = model(input_ids=ids, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        letter_ids = [tok.encode(letter, add_special_tokens=False) for letter in letters]
        assert all(len(x) == 1 for x in letter_ids)
        scores = logits[[x[0] for x in letter_ids]].log_softmax(-1).tolist()
        pred = int(np.argmax(scores))
        answer = row["choices"]["label"].index(row["answerKey"])
        records.append({"id": row["id"], "source_index": row["source_index"],
                        "question": row["question"], "choice_logprobs": scores,
                        "prediction": letters[pred], "answer": letters[answer], "correct": pred == answer})
    return records


def score_ifeval_batch(model, tok, batch, cfg):
    completions = generate(model, tok, batch, cfg["eval_max_new_tokens"], cfg, sample=False)
    for row, comp in zip(batch, completions):
        response = tok.decode(comp, skip_special_tokens=True)
        # Deterministic language detection for the language constraints.
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        metrics = process_results(row, [response])
        rec = {"id": row["id"], "prompt": row["prompt"], "response": response,
               "completion_tokens": len(comp), "truncated": comp[-1] != tok.eos_token_id, **metrics}
        yield rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["Teacher", "E0", "E1", "E2", "E3", "E4"], required=True)
    args = parser.parse_args()
    cfg = config()
    seed_all(cfg["seed"])
    out = Path("results") / args.name
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists():
        print(f"Already evaluated {args.name}")
        return
    path = cfg["teacher"] if args.name == "Teacher" else cfg["student"] if args.name == "E0" else f"runs/{args.name}/checkpoint"
    model = load_model(path, "cuda:0")
    tok = tokenizer(path)
    start = time.monotonic()
    rows = read_jsonl("data/ifeval.jsonl")
    file = out / "ifeval.jsonl"
    records = read_jsonl(file) if file.exists() else []
    done = {r["id"] for r in records}
    remaining = [row for row in rows if row["id"] not in done]
    # Greedy decoding and no truncation of benchmark prompts, for every model.
    with file.open("a") as f:
        for offset in range(0, len(remaining), cfg["eval_batch_size"]):
            batch = remaining[offset:offset+cfg["eval_batch_size"]]
            for rec in score_ifeval_batch(model, tok, batch, cfg):
                records.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({"name": args.name, "ifeval_done": len(records),
                              "seconds": time.monotonic()-start}), flush=True)
    assert len(records) == len(rows) == len({r["id"] for r in records})
    arc = evaluate_arc(model, tok, read_jsonl("data/arc.jsonl"))
    write_jsonl(out/"arc.jsonl", arc)
    instructions = [v for r in records for v in r["inst_level_strict_acc"]]
    result = {"name": args.name, "ifeval_prompts": len(records), "ifeval_instructions": len(instructions),
              "ifeval_strict_prompt_accuracy": float(np.mean([r["prompt_level_strict_acc"] for r in records])),
              "ifeval_strict_instruction_accuracy": float(np.mean(instructions)),
              "arc_accuracy": float(np.mean([r["correct"] for r in arc])), "arc_examples": len(arc),
              "ifeval_truncated": sum(r["truncated"] for r in records),
              "evaluation_seconds_this_session": time.monotonic()-start,
              "protocol": {"thinking": False, "ifeval_decoding": "greedy", "ifeval_max_tokens": cfg["eval_max_new_tokens"],
                           "arc": "zero-shot chat; argmax next-token A-E logprob; fixed seeded test subset"},
              **provenance()}
    write_json(out/"metrics.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
