"""Freeze dataset rows, non-thinking prompts, benchmark subset, and provenance."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

from kd.common import config, digest_file, prompt_ids, tokenizer, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--revisions", default="dataset_revisions.json")
    args = parser.parse_args()
    cfg = config(args.config)
    tok = tokenizer(cfg["student"])
    teacher_tok = tokenizer(cfg["teacher"])
    assert tok.get_vocab() == teacher_tok.get_vocab(), "KD requires identical token IDs"
    assert prompt_ids(tok, "Test") == prompt_ids(teacher_tok, "Test")
    names = ["HuggingFaceH4/ultrachat_200k", "google/IFEval", "allenai/ai2_arc"]
    revisions = json.loads(Path(args.revisions).read_text())
    assert set(names) <= revisions.keys(), "Missing pinned dataset revision"
    ds = load_dataset(names[0], split="train_sft", revision=revisions[names[0]])
    ifeval = load_dataset(names[1], split="train", revision=revisions[names[1]])
    arc = load_dataset(names[2], "ARC-Challenge", split="test", revision=revisions[names[2]])
    # Exclude exact normalized benchmark prompts/questions before sampling.
    norm = lambda x: " ".join(x.casefold().split())
    forbidden = {norm(r["prompt"]) for r in ifeval} | {norm(r["question"]) for r in arc}
    rng = random.Random(cfg["seed"])
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    seen, selected = set(), []
    excluded = {"duplicate": 0, "too_long": 0, "benchmark_overlap": 0, "empty": 0}
    for idx in indices:
        row = ds[idx]
        text = row["messages"][0]["content"]
        key = norm(text)
        if not key:
            excluded["empty"] += 1
            continue
        if key in forbidden:
            excluded["benchmark_overlap"] += 1
            continue
        if key in seen:
            excluded["duplicate"] += 1
            continue
        ids = prompt_ids(tok, text)
        if len(ids) > cfg["max_prompt_tokens"]:
            excluded["too_long"] += 1
            continue
        seen.add(key)
        selected.append({"id": hashlib.sha256(key.encode()).hexdigest(), "source_index": idx,
                         "prompt": text, "prompt_ids": ids})
        if len(selected) == cfg["train_size"] + cfg["validation_size"]:
            break
    assert len(selected) == cfg["train_size"] + cfg["validation_size"]
    train, valid = selected[:cfg["train_size"]], selected[cfg["train_size"]:]
    for split, rows in [("train", train), ("validation", valid)]:
        write_jsonl(f"data/{split}_prompts.jsonl", rows)
    eval_rows = [{**dict(row), "id": str(row["key"]), "prompt_ids": prompt_ids(tok, row["prompt"])}
                 for row in ifeval]
    write_jsonl("data/ifeval.jsonl", eval_rows)
    arc_ids = sorted(random.Random(cfg["seed"]).sample(range(len(arc)), cfg["arc_size"]))
    write_jsonl("data/arc.jsonl", [{**dict(arc[idx]), "source_index": idx} for idx in arc_ids])
    metadata = {"config": cfg, "dataset_revisions": revisions, "excluded_before_sample_complete": excluded,
                "train_count": len(train), "validation_count": len(valid), "ifeval_count": len(ifeval),
                "arc_count": len(arc_ids), "arc_indices": arc_ids,
                "selection": "seeded permutation of train_sft, unique first user turns fitting 512 tokens"}
    metadata["file_sha256"] = {str(p): digest_file(p) for p in sorted(Path("data").glob("*.jsonl"))}
    metadata["model_files_sha256"] = {
        kind: {p.name: digest_file(p) for p in sorted(Path(cfg[kind]).iterdir())
               if p.suffix in {".json", ".safetensors"}}
        for kind in ["student", "teacher"]}
    write_json("data/manifest.json", metadata)
    print({k: metadata[k] for k in ["train_count", "validation_count", "ifeval_count", "arc_count"]}, flush=True)


if __name__ == "__main__":
    main()
