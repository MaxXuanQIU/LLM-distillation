"""Compare the two archived training lengths on exactly matched benchmark items."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LONG = ROOT / "reproductions" / "completion-1024"
OUT = LONG / "results"
NAMES = ("Teacher", "E0", "E1", "E2", "E3", "E4")
LABELS = ("IFEval strict prompt", "IFEval strict instruction", "ARC-Challenge")


def rows(path):
    with path.open() as stream:
        return {row["id"]: row for row in map(json.loads, stream)}


def main():
    old_config, new_config = [json.loads((p / "config.json").read_text()) for p in (ROOT, LONG)]
    differences = {k: [v, new_config[k]] for k, v in old_config.items() if v != new_config[k]}
    assert differences == {"max_completion_tokens": [256, 1024]}, differences
    rng = np.random.default_rng(new_config["seed"])
    prompt_draw = rng.integers(541, size=(10000, 541))
    question_draw = rng.integers(400, size=(10000, 400))
    table = []
    for name in NAMES:
        before, after = [rows(p / "results" / name / "ifeval.jsonl") for p in (ROOT, LONG)]
        old_arc, new_arc = [rows(p / "results" / name / "arc.jsonl") for p in (ROOT, LONG)]
        assert before.keys() == after.keys() and len(after) == 541
        assert old_arc.keys() == new_arc.keys() and len(new_arc) == 400
        ids = sorted(before)
        arc_ids = sorted(old_arc)
        counts = np.array([len(before[i]["inst_level_strict_acc"]) for i in ids])
        assert all(len(after[i]["inst_level_strict_acc"]) == n for i, n in zip(ids, counts))
        for metric, label in enumerate(LABELS):
            if metric == 0:
                old, new = [np.array([r[i]["prompt_level_strict_acc"] for i in ids], dtype=float) for r in (before, after)]
                denom, draw = np.ones(len(ids)), prompt_draw
            elif metric == 1:
                old, new = [np.array([sum(r[i]["inst_level_strict_acc"]) for i in ids]) for r in (before, after)]
                denom, draw = counts, prompt_draw
            else:
                old, new = [np.array([r[i]["correct"] for i in arc_ids], dtype=float) for r in (old_arc, new_arc)]
                denom, draw = np.ones(len(arc_ids)), question_draw
            delta = new - old
            ci = np.quantile(100 * delta[draw].sum(axis=1) / denom[draw].sum(axis=1), [.025, .975])
            table.append(dict(model=name, metric=label, short_pct=100*old.sum()/denom.sum(),
                              long_pct=100*new.sum()/denom.sum(), delta_pp=100*delta.sum()/denom.sum(),
                              low95_pp=ci[0], high95_pp=ci[1]))
    with (OUT / "length_comparison.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained", sharey=True)
    for ax, label in zip(axes, LABELS):
        records = [r for r in table if r["metric"] == label and r["model"] in NAMES[2:]]
        x = np.array([r["delta_pp"] for r in records])
        lo = np.array([r["low95_pp"] for r in records])
        hi = np.array([r["high95_pp"] for r in records])
        ax.axvline(0, color="gray", lw=1)
        ax.errorbar(x, range(4), xerr=[x-lo, hi-x], fmt="o", capsize=3)
        ax.set(title=label, xlabel="1,024 minus 256 tokens (pp)", yticks=range(4),
               yticklabels=["Hard-label SFT", "Off-policy KL", "Hybrid KL", "On-policy KL"])
    axes[0].invert_yaxis()
    fig.savefig(OUT / "figures" / "length_comparison.png", dpi=170)
    plt.close(fig)
    prefix_stats = {}
    for split in ("train", "validation"):
        short, long = [rows(p / "data" / f"{split}.jsonl") for p in (ROOT, LONG)]
        assert short.keys() == long.keys()
        assert all(short[i]["prompt_ids"] == long[i]["prompt_ids"] for i in short)
        prefix_stats[split] = dict(examples=len(short),
            matching_short_response_prefix=sum(long[i]["completion_ids"][:len(r["completion_ids"])] == r["completion_ids"]
                                                for i, r in short.items()),
            capped_256=sum(r["truncated"] for r in short.values()),
            capped_1024=sum(r["truncated"] for r in long.values()),
            tokens_256=sum(len(r["completion_ids"]) for r in short.values()),
            tokens_1024=sum(len(r["completion_ids"]) for r in long.values()))
    (OUT / "length_protocol.json").write_text(json.dumps(dict(config_changes=differences,
        fixed_data=prefix_stats, paired_bootstrap_replicates=10000, seed=new_config["seed"]), indent=2)+"\n")


if __name__ == "__main__":
    main()
