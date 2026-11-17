"""Check experimental comparability and artifact completeness from saved records."""
import argparse
import json
import math
from pathlib import Path

from kd.common import config, digest_file, read_jsonl, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-only", action="store_true")
    args = p.parse_args()
    cfg = config()
    train, valid = read_jsonl("data/train.jsonl"), read_jsonl("data/validation.jsonl")
    assert len(train) == cfg["train_size"] and len(valid) == cfg["validation_size"]
    assert len({r["id"] for r in train+valid}) == len(train)+len(valid)
    assert all(0 < len(r["completion_ids"]) <= cfg["max_completion_tokens"] for r in train+valid)
    assert all(0 < len(r["prompt_ids"]) <= cfg["max_prompt_tokens"] for r in train+valid)
    fixed = {r["id"]: r["completion_ids"] for r in train}
    checks = {"train_count": len(train), "validation_count": len(valid), "disjoint_unique_prompts": True,
              "fixed_train_completion_tokens": sum(map(len, fixed.values())),
              "teacher_train_truncated": sum(r["truncated"] for r in train),
              "data_sha256": {str(p): digest_file(p) for p in sorted(Path("data").glob("*.jsonl"))}}
    manifest = json.loads(Path("data/manifest.json").read_text())
    for path, expected in manifest["file_sha256"].items():
        assert digest_file(path) == expected, f"Prepared data changed: {path}"
    checks["prepared_data_unchanged"] = True
    if not args.data_only:
        initial_audit = json.loads(Path("data/audit.json").read_text())
        assert initial_audit["data_sha256"] == checks["data_sha256"], "Fixed teacher dataset changed after preparation"
        checks["fixed_teacher_dataset_unchanged"] = True
        for kind, files in manifest["model_files_sha256"].items():
            for filename, expected in files.items():
                assert digest_file(Path(cfg[kind])/filename) == expected, f"Source model changed: {kind}/{filename}"
        checks["original_model_files_unchanged"] = True
        checks["experiments"] = {}
        configs = []
        run_revisions = set()
        for n in ["E1", "E2", "E3", "E4"]:
            root = Path("runs") / n
            summary = json.loads((root/"summary.json").read_text())
            meta = json.loads((root/"metadata.json").read_text())
            trajectory = read_jsonl(root/"trajectories.jsonl")
            configs.append(meta["config"])
            run_revisions.add(meta["git_commit"])
            assert meta["config"] == cfg
            assert len(trajectory) == len(train) == len({r["id"] for r in trajectory})
            assert {r["id"] for r in trajectory} == set(fixed)
            assert summary["optimizer_steps"] == math.ceil(len(train)/cfg["effective_batch_size"])
            assert summary["completion_tokens"] == sum(len(r["completion_ids"]) for r in trajectory)
            assert summary["on_policy_examples"] == sum(r["on_policy"] for r in trajectory)
            assert all(r["completion_ids"] == fixed[r["id"]] for r in trajectory if not r["on_policy"])
            if n in ["E1", "E2"]:
                assert not any(r["on_policy"] for r in trajectory)
            elif n == "E4":
                assert all(r["on_policy"] for r in trajectory)
            weights = list((root/"checkpoint").glob("*.safetensors"))
            assert weights
            checks["experiments"][n] = {"examples": len(trajectory),
                "on_policy_examples": summary["on_policy_examples"],
                "truncated_completions": sum(r["truncated"] for r in trajectory),
                "completion_tokens": summary["completion_tokens"],
                "checkpoint_sha256": {p.name: digest_file(p) for p in weights}}
        for n in ["Teacher", "E0", "E1", "E2", "E3", "E4"]:
            metrics = json.loads(Path(f"results/{n}/metrics.json").read_text())
            run_revisions.add(metrics["git_commit"])
            for benchmark, expected in [("ifeval", 541), ("arc", cfg["arc_size"])]:
                records = read_jsonl(f"results/{n}/{benchmark}.jsonl")
                assert len(records) == expected == len({r["id"] for r in records})
                assert {r["id"] for r in records} == {r["id"] for r in read_jsonl(f"data/{benchmark}.jsonl")}
                if benchmark == "ifeval":
                    prompt_acc = sum(r["prompt_level_strict_acc"] for r in records)/len(records)
                    instructions = [v for r in records for v in r["inst_level_strict_acc"]]
                    assert abs(prompt_acc-metrics["ifeval_strict_prompt_accuracy"]) < 1e-12
                    assert abs(sum(instructions)/len(instructions)-metrics["ifeval_strict_instruction_accuracy"]) < 1e-12
                else:
                    assert abs(sum(r["correct"] for r in records)/len(records)-metrics["arc_accuracy"]) < 1e-12
        assert len(run_revisions) == 1, "Model runs used different code revisions"
        checks["frozen_model_run_revision"] = next(iter(run_revisions))
    write_json("data/audit.json" if args.data_only else "results/audit.json", checks)
    print(json.dumps({"audit": "passed", "data_only": args.data_only, "train_count": len(train)}))


if __name__ == "__main__":
    main()
