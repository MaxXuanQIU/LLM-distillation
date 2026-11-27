"""Summarize audited long-completion benchmark contrasts and training costs."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads((root / path).read_text())


def delta(pair, metric):
    return 100 * comparisons[pair][metric]["delta"]


prompt = "ifeval_strict_prompt_accuracy"
instruction = "ifeval_strict_instruction_accuracy"
arc = "arc_accuracy"
scores = {n: load(f"results/{n}/metrics.json") for n in ("Teacher", "E0", "E1", "E2", "E3", "E4")}
comparisons = load("results/comparisons.json")["comparisons"]
runs = {n: load(f"runs/{n}/summary.json") for n in ("E1", "E2", "E3", "E4")}
audit = load("results/audit.json")
baseline_ce = load("results/E0/validation.json")["validation_cross_entropy"]
cap = load("config.json")["max_completion_tokens"]
negative = sum(comparisons[f"{n}-E0"][k]["ci95"][1] < 0 for n in runs for k in (prompt, instruction, arc))
lower_ce = sum(t["validation_cross_entropy"] < baseline_ce for t in runs.values())
pairs = ("E2-E1", "E4-E2", "E3-E2", "E3-E4")
overlap = sum(v["ci95"][0] <= 0 <= v["ci95"][1] for pair in pairs for v in comparisons[pair].values())
e2, e3, e4 = (runs[n] for n in ("E2", "E3", "E4"))
fixed_tokens = audit["fixed_train_completion_tokens"]
capped = audit["experiments"]["E4"]["truncated_completions"]

lines = ["## Long-completion results", ""]
if negative == 12:
    lines.append("The original student outscored every trained checkpoint on strict IFEval prompt accuracy, "
                 "strict instruction accuracy and ARC-Challenge. All 12 paired 95% intervals for the changes "
                 "from baseline fell below zero.")
else:
    lines.append(f"Of 12 paired 95% intervals for changes from the original student, {negative} fell below zero.")
lines.extend([
    ("All four methods" if lower_ce == 4 else f"{lower_ce} of the four methods") + " reduced held-out loss on teacher responses. "
    "This measures how closely a student predicts the teacher on that validation set.", "",
    f"**Hard and soft labels.** On exactly the same {fixed_tokens:,} fixed response tokens, replacing hard labels "
    f"with forward KL changed strict prompt accuracy by {delta('E2-E1',prompt):+.2f} percentage points, "
    f"strict instruction accuracy by {delta('E2-E1',instruction):+.2f} points, and ARC by {delta('E2-E1',arc):+.2f} points.", "",
    f"**Student-generated training contexts.** Moving from fully off-policy to fully on-policy KL changed strict "
    f"instruction accuracy by {delta('E4-E2',instruction):+.2f} points and ARC by {delta('E4-E2',arc):+.2f} points. "
    f"On-policy KL consumed {e4['allocated_gpu_hours']:.2f} allocated training GPU-hours compared with "
    f"{e2['allocated_gpu_hours']:.2f} for off-policy KL. The shared teacher-data generation cost is reported separately.", "",
    f"**Hybrid training.** The hybrid used fresh student answers for {e3['on_policy_examples']:,} of "
    f"{e3['examples']:,} prompts. Its strict instruction accuracy changed by {delta('E3-E2',instruction):+.2f} points "
    f"relative to off-policy KL and {delta('E3-E4',instruction):+.2f} relative to on-policy KL. "
    f"The corresponding ARC differences were {delta('E3-E2',arc):+.2f} and {delta('E3-E4',arc):+.2f} points. "
    f"Across the four direct comparisons between training methods, {overlap} of 12 paired 95% intervals included zero.", "",
    f"**Response length.** With a {cap:,}-token ceiling, {audit['teacher_train_truncated']:,} of "
    f"{audit['train_count']:,} fixed teacher answers reached the limit. In on-policy training, "
    f"{capped:,} answers reached it and contributed {100*cap*capped/e4['completion_tokens']:.1f}% of response tokens. "
    f"On IFEval, the original student reached the ceiling on {scores['E0']['ifeval_truncated']} prompts. "
    f"Counts for SFT, off-policy, hybrid and on-policy training were {scores['E1']['ifeval_truncated']}, "
    f"{scores['E2']['ifeval_truncated']}, {scores['E3']['ifeval_truncated']} and {scores['E4']['ifeval_truncated']}. "
    "Prompt budgets and optimizer steps were matched. Changing the training contexts also changed the token budget and compute.", "",
    "Each condition was trained once. Unadjusted paired intervals resample evaluation items and quantify "
    "uncertainty for these checkpoints. Variation across training seeds and the effect of increasing the ceiling "
    "past the studied value remain open.", "",
])
(root / "docs/conclusions.md").write_text("\n".join(lines))
