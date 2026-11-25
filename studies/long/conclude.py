"""Write conclusions for the long run from its audited scores and paired comparisons."""
from pathlib import Path
import json

root = Path(__file__).resolve().parents[2]
def load(path):
    return json.loads((root / path).read_text())

metrics = {n:load(f"results/{n}/metrics.json") for n in ["Teacher", "E0", "E1", "E2", "E3", "E4"]}
comparisons = load("results/comparisons.json")["comparisons"]
audit = load("results/audit.json")
cfg = load("config.json")
trajectories = audit["experiments"]
methods = {"E1":"Hard-label SFT", "E2":"Off-policy KL", "E3":"Hybrid KL", "E4":"On-policy KL"}
instruction = "ifeval_strict_instruction_accuracy"
arc = "arc_accuracy"
lines = ["## Long-completion results", "",
         f"The teacher generated {audit['train_count']:,} training answers with a {cfg['max_completion_tokens']:,}-token ceiling. "
         f"{audit['teacher_train_truncated']:,} ({100*audit['teacher_train_truncated']/audit['train_count']:.2f}%) reached the ceiling. "
         f"The fixed training set contained {audit['fixed_train_completion_tokens']:,} response tokens.", ""]
for n, method in methods.items():
    delta = comparisons[f"{n}-E0"]
    lines.append(f"{method} scored {100*metrics[n][instruction]:.2f}% on strict instruction accuracy "
                 f"({100*delta[instruction]['delta']:+.2f} points relative to the original student) "
                 f"and {100*metrics[n][arc]:.2f}% on ARC ({100*delta[arc]['delta']:+.2f} points).")
lines.extend(["", "The following matched comparisons use unadjusted 95% bootstrap intervals on the benchmark items.", ""])
for pair, label in [("E2-E1","Soft labels versus hard labels"), ("E4-E2","On-policy versus off-policy KL"),
                    ("E3-E2","Hybrid versus off-policy KL"), ("E3-E4","Hybrid versus on-policy KL")]:
    results = []
    for k, desc in [(instruction,"strict instruction accuracy"), (arc,"ARC")]:
        val = comparisons[pair][k]
        lo,hi = [100*x for x in val["ci95"]]
        direction = "higher" if lo > 0 else "lower" if hi < 0 else "unresolved"
        results.append(f"{desc} {100*val['delta']:+.2f} pp (95% interval {lo:+.2f} to {hi:+.2f}, {direction})")
    lines.extend([f"**{label}.** " + " and ".join(results) + ".", ""])
lines.extend(["The uncertainty intervals measure variability across evaluation items. Each method was trained once. "
              "The token budget, full-parameter update, sampled UltraChat prompts and zero-shot ARC letter scoring define the scope of the findings.", ""])
(root/"docs/conclusions.md").write_text("\n".join(lines))
