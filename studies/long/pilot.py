"""Estimate length censoring before training, using a fixed validation sample."""
import argparse
import random
from kd.common import config, generate, load_model, read_jsonl, seed_all, tokenizer, write_json

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["teacher", "student"])
parser.add_argument("shard", type=int, choices=[0, 1])
args = parser.parse_args()
cfg = config()
seed_all(cfg["seed"] + args.shard)
rows = random.Random(cfg["seed"]).sample(read_jsonl("data/validation_prompts.jsonl"), 64)[args.shard::2]
model = load_model(cfg[args.model], "cuda:0")
tok = tokenizer(cfg[args.model])
result = {}
for start in range(0, len(rows), 16):
    batch = rows[start:start+16]
    for row, completion in zip(batch, generate(model, tok, batch, cfg["max_completion_tokens"], cfg)):
        result[row["id"]] = {"tokens":len(completion), "ended_with_eos":completion[-1] == tok.eos_token_id}
    write_json(f"long_pilot/{args.model}_{args.shard}.json", result)
