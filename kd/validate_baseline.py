"""Measure original-student held-out CE in the training validation precision."""
import time
from pathlib import Path

from kd.common import config, load_model, provenance, read_jsonl, seed_all, tokenizer, write_json
from kd.train import validation


def main():
    output = Path("results/E0/validation.json")
    if output.exists():
        return
    cfg = config()
    seed_all(cfg["seed"])
    model = load_model(cfg["student"], "cuda:0", train=True)
    tok = tokenizer(cfg["student"])
    rows = read_jsonl("data/validation.jsonl")
    start = time.monotonic()
    ce = validation(model, tok, rows, cfg)
    write_json(output, {"name": "E0", "validation_examples": len(rows),
               "validation_cross_entropy": ce, "seconds": time.monotonic()-start,
               "precision": "FP32 parameters and BF16 autocast, matching training validation", **provenance()})


if __name__ == "__main__":
    main()
