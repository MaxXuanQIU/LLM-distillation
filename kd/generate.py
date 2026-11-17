import argparse
import json
import time
from pathlib import Path

import torch

from kd.common import config, generate, load_model, read_jsonl, seed_all, tokenizer, write_json, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--merge", action="store_true")
    args = p.parse_args()
    cfg = config()
    if args.merge:
        generated = [row for i in range(args.shards) for row in read_jsonl(f"data/teacher_{i}.jsonl")]
        by_id = {r["id"]: r for r in generated}
        assert len(by_id) == len(generated), "Duplicate generation IDs"
        for split in ["train", "validation"]:
            prompts = read_jsonl(f"data/{split}_prompts.jsonl")
            rows = [by_id[r["id"]] for r in prompts]
            assert all(0 < len(r["completion_ids"]) <= cfg["max_completion_tokens"] for r in rows)
            write_jsonl(f"data/{split}.jsonl", rows)
        return
    seed_all(cfg["seed"] + args.shard)
    rows = read_jsonl("data/train_prompts.jsonl") + read_jsonl("data/validation_prompts.jsonl")
    rows = rows[args.shard::args.shards]
    out = Path(f"data/teacher_{args.shard}.jsonl")
    done = {r["id"] for r in read_jsonl(out)} if out.exists() else set()
    rows = [r for r in rows if r["id"] not in done]
    model = load_model(cfg["teacher"], "cuda:0")
    tok = tokenizer(cfg["teacher"])
    bs = cfg["generation_batch_size"]
    start = time.monotonic()
    count = 0
    with out.open("a") as f:
        for offset in range(0, len(rows), bs):
            # Per-batch seeds make clean restarts reproducible at batch boundaries.
            batch = rows[offset:offset+bs]
            seed_all(cfg["seed"] + int(batch[0]["id"][:8], 16) % 100000000)
            comps = generate(model, tok, batch, cfg["max_completion_tokens"], cfg)
            for row, comp in zip(batch, comps):
                f.write(json.dumps({**row, "completion_ids": comp,
                    "response": tok.decode(comp, skip_special_tokens=True),
                    "truncated": comp[-1] != tok.eos_token_id}, ensure_ascii=False) + "\n")
                count += len(comp)
            f.flush()
            print(json.dumps({"shard": args.shard, "done": offset+len(batch)+len(done),
                              "tokens_per_second": count/(time.monotonic()-start)}), flush=True)
    write_json(f"data/teacher_{args.shard}_timing.json", {
        "seconds": time.monotonic()-start, "new_tokens": count,
        "peak_memory_gb": torch.cuda.max_memory_allocated()/1e9})


if __name__ == "__main__":
    main()
