"""Generate the results table, paired uncertainty estimates, and figures."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kd.common import config, digest_file, read_jsonl, write_json
from kd.accounting import summarize


def main():
    cfg = config()
    names = ["Teacher", "E0", "E1", "E2", "E3", "E4"]
    metrics = {n: json.loads(Path(f"results/{n}/metrics.json").read_text()) for n in names}
    train = {n: json.loads(Path(f"runs/{n}/summary.json").read_text()) for n in names[2:]}
    histories = {n: json.loads(Path(f"runs/{n}/history.json").read_text()) for n in names[2:]}
    baseline_val = json.loads(Path("results/E0/validation.json").read_text())
    generation = [json.loads(p.read_text()) for p in sorted(Path("data").glob("teacher_*_timing.json"))]
    generation_gpu_hours = sum(r["seconds"] for r in generation)/3600
    methods = {"Teacher": "Qwen3-4B", "E0": "Original Qwen3-0.6B", "E1": "Hard-label SFT",
               "E2": "Off-policy KL (λ=0)", "E3": "Hybrid KL (λ=0.5)", "E4": "On-policy KL (λ=1)"}
    ifeval = {n: sorted(read_jsonl(f"results/{n}/ifeval.jsonl"), key=lambda r: r["id"]) for n in names}
    arc = {n: sorted(read_jsonl(f"results/{n}/arc.jsonl"), key=lambda r: r["id"]) for n in names}
    for source in [ifeval, arc]:
        assert all([r["id"] for r in source[n]] == [r["id"] for r in source["E0"]] for n in names)
    timing, length_stats = summarize(metrics, train, histories, ifeval, baseline_val, generation)
    diagnostics = json.loads(Path("results/training_diagnostics.json").read_text())
    counts = np.array([len(r["inst_level_strict_acc"]) for r in ifeval["E0"]])
    rng = np.random.default_rng(cfg["seed"])
    ix = rng.integers(0, len(counts), (10000, len(counts)))
    ax = rng.integers(0, len(arc["E0"]), (10000, len(arc["E0"])))
    samples, intervals = {}, {}
    keys = ["ifeval_strict_prompt_accuracy", "ifeval_strict_instruction_accuracy", "arc_accuracy"]
    for n in names:
        prompt = np.array([r["prompt_level_strict_acc"] for r in ifeval[n]], dtype=float)
        inst = np.array([sum(r["inst_level_strict_acc"]) for r in ifeval[n]])
        correct = np.array([r["correct"] for r in arc[n]], dtype=float)
        samples[n] = {keys[0]: prompt[ix].mean(1), keys[1]: inst[ix].sum(1)/counts[ix].sum(1),
                      keys[2]: correct[ax].mean(1)}
        intervals[n] = {k: np.quantile(samples[n][k], [.025, .975]).tolist() for k in keys}
    pairs = [("E1", "E0"), ("E2", "E0"), ("E3", "E0"), ("E4", "E0"),
             ("E2", "E1"), ("E4", "E2"), ("E3", "E2"), ("E3", "E4")]
    comparisons = {f"{a}-{b}": {k: {"delta": metrics[a][k]-metrics[b][k],
                   "ci95": np.quantile(samples[a][k]-samples[b][k], [.025, .975]).tolist()}
                   for k in keys} for a, b in pairs}
    write_json("results/comparisons.json", {"bootstrap_replicates": 10000,
                "method": "paired bootstrap; resample prompts for IFEval, questions for ARC; unadjusted intervals",
                "intervals": intervals, "comparisons": comparisons})
    with open("results/summary.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "method", *keys, "completion_tokens", "allocated_gpu_hours", "validation_ce"], lineterminator="\n")
        writer.writeheader()
        for n in names:
            row = {"model": n, "method": methods[n], **{k: metrics[n][k] for k in keys}}
            if n == "E0":
                row["validation_ce"] = baseline_val["validation_cross_entropy"]
            if n in train:
                row.update(completion_tokens=train[n]["completion_tokens"],
                           allocated_gpu_hours=train[n]["allocated_gpu_hours"],
                           validation_ce=train[n]["validation_cross_entropy"])
            writer.writerow(row)
    figures = Path("results/figures")
    figures.mkdir(exist_ok=True)
    colors = ["#757575", "#222222", "#4e79a7", "#f28e2b", "#59a14f", "#b07aa1"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for ax, k, title in zip(axes, keys, ["IFEval: strict prompt", "IFEval: strict instruction", "ARC-Challenge"]):
        values = np.array([metrics[n][k]*100 for n in names])
        ci = np.array([intervals[n][k] for n in names]).T*100
        ax.bar(names, values, color=colors, yerr=[values-ci[0], ci[1]-values], capsize=3)
        ax.set(title=title, ylabel="Accuracy (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=35)
    fig.savefig(figures/"benchmark_results.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, k, title in zip(axes, keys[1:], ["IFEval: strict instruction", "ARC-Challenge"]):
        selected = ["E2", "E3", "E4"]
        values = np.array([metrics[n][k]*100 for n in selected])
        ci = np.array([intervals[n][k] for n in selected]).T*100
        ax.errorbar([0, .5, 1], values, yerr=[values-ci[0], ci[1]-values], marker="o", capsize=4)
        ax.axhline(metrics["E0"][k]*100, color="#777777", linestyle="--", label="Original student")
        ax.set(title=title, xlabel="On-policy probability λ", ylabel="Accuracy (%)", xticks=[0, .5, 1])
        ax.legend()
    fig.savefig(figures/"on_policy_ratio.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for n, color in zip(names[2:], colors[2:]):
        h = histories[n]
        ax = axes[0] if n == "E1" else axes[1]
        y = np.array([r["loss"] for r in h])
        ax.plot([r["step"] for r in h], y, alpha=.2, color=color)
        window = min(15, len(y))
        ax.plot([r["step"] for r in h][window-1:], np.convolve(y, np.ones(window)/window, mode="valid"), label=n, color=color)
    for ax, title in zip(axes, ["E1: hard-label cross entropy", "E2–E4: forward KL on training contexts"]):
        ax.set(title=title, xlabel="Optimizer step", ylabel="Loss per response token (nats)")
        ax.legend()
    fig.savefig(figures/"training_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    axes[0].bar(names[2:], [train[n]["allocated_gpu_hours"] for n in names[2:]], color=colors[2:])
    axes[0].set(ylabel="Allocated training GPU hours", title="Compute cost")
    axes[1].bar(names[2:], [train[n]["completion_tokens"]/1e6 for n in names[2:]], color=colors[2:])
    axes[1].set(ylabel="Million supervised response tokens", title="Actual token budgets")
    fig.savefig(figures/"training_efficiency.png", dpi=180)
    plt.close(fig)

    lines = ["# Off-policy, on-policy, and hybrid Qwen3 distillation", "", "## Completed experiment", "",
        "One full-parameter training run per condition, using Qwen3-4B as a frozen teacher and the same original Qwen3-0.6B initialization. "
        "All models use non-thinking mode. These results describe this dataset, budget, and training seed; they do not establish a general ranking of distillation methods.", ""]
    conclusions = Path("docs/conclusions.md")
    if conclusions.exists():
        lines += [conclusions.read_text().strip(), ""]
    lines += ["## Results", "", "| Model | IFEval strict prompt | IFEval strict instruction | ARC-Challenge |",
        "|---|---:|---:|---:|"]
    for n in names:
        values = [f"{metrics[n][k]*100:.2f}% [{intervals[n][k][0]*100:.2f}, {intervals[n][k][1]*100:.2f}]" for k in keys]
        lines.append(f"| {n}: {methods[n]} | " + " | ".join(values) + " |")
    lines += ["", "Brackets show 95% bootstrap intervals, resampling complete prompts for IFEval and questions for ARC. "
              "The instruction metric is micro-averaged. These intervals measure evaluation-set uncertainty, not training-seed variability.",
              "", "![Benchmark results](figures/benchmark_results.png)", "",
              "Changes relative to E0, in percentage points:", "",
              "| Model | Δ strict prompt | Δ strict instruction | Δ ARC |", "|---|---:|---:|---:|"]
    for n in names[2:]:
        lines.append(f"| {n} | " + " | ".join(f"{(metrics[n][k]-metrics['E0'][k])*100:+.2f}" for k in keys) + " |")
    lines += ["", "## Research questions", ""]
    for label, pair in [("RQ1: soft vs hard labels", "E2-E1"), ("RQ2: on-policy vs off-policy", "E4-E2"),
                        ("RQ3: hybrid vs off-policy", "E3-E2"), ("RQ3: hybrid vs on-policy", "E3-E4")]:
        c = comparisons[pair]
        lines.append(f"**{label} ({pair}).** " + "; ".join(
            f"{k.replace('_accuracy', '').replace('_', ' ')}: {c[k]['delta']*100:+.2f} percentage points "
            f"(paired 95% CI {c[k]['ci95'][0]*100:+.2f} to {c[k]['ci95'][1]*100:+.2f})" for k in keys) + ".")
        lines.append("")
    lines += ["These are exploratory, unadjusted pairwise intervals. An interval spanning zero does not resolve the sign of the difference on this evaluation set. "
              "Multiple metrics/comparisons and the single training seed limit hypothesis claims.", "", "## Training budget and behavior", "",
              "| Run | λ | Steps | Response tokens | Realized on-policy fraction | Allocated GPU hours | Validation CE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for n, t in train.items():
        lmbda = "—" if n == "E1" else t["lambda"]
        lines.append(f"| {n} | {lmbda} | {t['optimizer_steps']} | {t['completion_tokens']:,} | "
                     f"{t['on_policy_examples']/t['examples']:.3f} | {t['allocated_gpu_hours']:.3f} | {t['validation_cross_entropy']:.4f} |")
    if "main_phase_wall_hours" in timing:
        lines += ["", f"The complete main phase took {timing['main_phase_wall_hours']:.3f} wall-clock hours and "
                  f"{timing['main_phase_job_gpu_hours']:.3f} allocated job GPU hours, including loading, validation and export. "
                  "It excludes preliminary pilots, shared teacher-data preparation, and the separately measured baseline validation. "
                  f"The recorded job schedule peaks at {timing['peak_concurrent_assigned_gpus']} concurrent GPUs, all in physical IDs 0–3.", ""]
    lines += ["", "GPU hours are wall time × allocated devices, including student generation and idle teacher time; they are not integrated SM activity or energy. "
              "Teacher dataset generation and benchmark evaluation are separate costs. Validation CE uses the same held-out teacher sequences for every run.",
              "", f"The original student's validation CE is {baseline_val['validation_cross_entropy']:.4f}. "
              "Lower held-out teacher-response loss does not necessarily imply higher benchmark accuracy.",
              "", f"Shared teacher-data generation loops used {generation_gpu_hours:.3f} GPU hours for "
              f"{sum(r['new_tokens'] for r in generation):,} tokens across 5,200 responses. "
              "This cost is excluded from the training bars. E4 does not use fixed training responses, so charging the entire shared preprocessing cost to each method would misrepresent standalone costs. "
              "All runs use the held-out teacher responses for validation.",
              "", f"The four training loops used {timing['training_loop_gpu_hours']:.3f} allocated GPU hours in total; "
              f"the six evaluation loops used {timing['evaluation_loop_gpu_hours']:.3f}. "
              "Training timers include periodic checkpoint writes but exclude initial loading, final validation, and final model export. "
              "Generation and evaluation timers exclude initial model loading. Student-rollout generation is already included in training time.",
              "", "![On-policy ratio](figures/on_policy_ratio.png)",
              "", "![Training curves](figures/training_curves.png)", "", "![Training efficiency](figures/training_efficiency.png)",
              "", "KL losses across E2–E4 are measured on different generated contexts and cannot alone rank model quality. E1 cross entropy is a different objective.",
              "", "Mean training loss over the first 20 versus last 20 optimizer steps: " + "; ".join(
                  f"{n} {np.mean([r['loss'] for r in histories[n][:20]]):.4f} → "
                  f"{np.mean([r['loss'] for r in histories[n][-20:]]):.4f}" for n in names[2:]) +
              ". These averages describe within-run optimization; they do not demonstrate convergence to an optimum.",
              "", f"Student rollout generation consumed {100*train['E3']['generation_seconds']/train['E3']['training_seconds']:.1f}% "
              f"of hybrid training wall time and {100*train['E4']['generation_seconds']/train['E4']['training_seconds']:.1f}% "
              "of on-policy training wall time. The completion cap was reached by "
              f"{diagnostics['E3']['training_capped_responses']:,}/{train['E3']['examples']:,} hybrid sequences and "
              f"{diagnostics['E4']['training_capped_responses']:,}/{train['E4']['examples']:,} on-policy sequences.",
              "", "## Protocol", "", "```json", json.dumps(cfg, indent=2), "```", "",
              "UltraChat train_sft rows were randomly permuted with the recorded seed. We selected unique first user prompts fitting the 512-token chat limit, "
              "excluding exact normalized benchmark-question matches, then split 5,000 training and 200 validation prompts. This length filter changes the source distribution; "
              "the exact-match screen cannot exclude paraphrased contamination. Teacher responses were generated once and reused without retokenizing.", "",
              "E1 uses completion-token cross entropy. E2–E4 use exact full-vocabulary KL(teacher || student) at temperature 1, with β=0. "
              "E3 chooses the source independently per microbatch with probability 0.5. E4 generates from the current student at every microbatch; "
              "sampling occurs without gradients and the resulting token IDs are treated as fixed contexts. No adapters, rewards, or preference data are used.", "",
              "Training uses FP32 master parameters and AdamW states with BF16 autocast, gradient checkpointing, response-only masking, "
              "token-weighted effective batches, gradient clipping, and a cosine learning-rate schedule. Every condition sees one epoch and the same optimizer-step budget. "
              "Actual token budgets differ because completions can terminate early. No benchmark scores were used to select hyperparameters.", "",
              "IFEval: all 541 prompts, greedy decoding, a 2,048-token completion limit, and lm-evaluation-harness 0.4.9.2 strict scoring. "
              "ARC: a seeded 400-question test subset, zero-shot chat with choices and an instruction to answer with a letter, "
              "then argmax over the single-token letter likelihoods. This custom protocol must not be compared directly with default harness ARC leaderboard results.", "",
              "## Truncation and limitations", ""]
    lines += ["| Model | Mean response tokens | Median | 95th percentile | Capped IFEval responses |",
              "|---|---:|---:|---:|---:|"]
    for n in names:
        s = length_stats[n]
        lines.append(f"| {n} | {s['mean_tokens']:.1f} | {s['median_tokens']:.0f} | {s['p95_tokens']:.0f} | {s['capped']}/541 |")
    fixed = read_jsonl("data/train.jsonl")
    lines += ["", f"Teacher training responses reaching the completion cap: {sum(r['truncated'] for r in fixed)}/{len(fixed)} "
              f"({100*sum(r['truncated'] for r in fixed)/len(fixed):.2f}%). "
              "A truncated completion is supervised as observed, without inventing an EOS token. This can teach incomplete answers and is a material limitation of the proposed short-completion budget.", "",
              "The student already has instruction tuning. Only one training seed, one modest dataset, and two benchmarks were studied. "
              "The source mix changes context distributions, response lengths, and compute costs. Paired evaluation intervals do not remove these limitations.",
              "", "## Selected generations", "",
              "Improvement and regression refer to automatic strict instruction checks. Passing those checks does not establish factual accuracy, fluency, or usefulness; the E1 example at key 1082 passes its format checks despite substantial repetition.", ""]
    baseline = {r["id"]: r for r in ifeval["E0"]}
    for n in names[2:]:
        for category, predicate in [("improvement", lambda b, r: not b["prompt_level_strict_acc"] and r["prompt_level_strict_acc"]),
                                    ("regression", lambda b, r: b["prompt_level_strict_acc"] and not r["prompt_level_strict_acc"])]:
            candidates = [r for r in ifeval[n] if predicate(baseline[r["id"]], r)]
            if candidates:
                r = candidates[0]
                lines += [f"### {n}: {category}, IFEval key {r['id']}", "", "Prompt:", "", "```text", r["prompt"], "```", "",
                          "E0 response (excerpt):", "", "```text", baseline[r["id"]]["response"][:1800], "```", "",
                          f"{n} response (excerpt):", "", "```text", r["response"][:1800], "```", ""]
    lines += ["Examples are selected deterministically by the first lexicographically sorted benchmark key meeting the stated change in strict prompt success. "
              "Excerpts may be shortened and trailing spaces omitted for display; scoring used unmodified complete responses. Full records remain in the per-model JSONL files.", "",
              "## Artifacts and reproducibility", "",
              "The server checkout is `/home/data2/xq/projects/LLM-distillation` on `10.123.4.22`. "
              "A local copy of the fixed dataset, all four full checkpoints, and evaluation records is also retained in `/home/xq/projects/LLM-distillation`. "
              "`data/` contains fixed prompts, teacher token sequences, benchmark rows, and hashes; `runs/E*/checkpoint/` contains the four full student checkpoints; "
              "`runs/E*/history.json` contains training traces; `results/<model>/` contains per-example outputs and metrics. "
              "`results/summary.csv` and `results/comparisons.json` contain machine-readable summaries. The code and run provenance record Git revisions and package versions.", "",
              f"All model runs used frozen revision `{metrics['E0']['git_commit']}`. Later changes add analysis, auditing, and reproduction helpers. "
              "`dataset_revisions.json` pins the source data; `results/data_manifest.json` preserves the original preparation manifest. "
              "An isolated preparation rerun reproduced all four prompt/benchmark file hashes exactly (`results/reproduction_check.json`). "
              "The final artifact audit verifies source models, fixed teacher data, training coverage, benchmark identities, and metric aggregation.", "",
              "References: [On-policy distillation / GKD](https://arxiv.org/abs/2306.13649), "
              "[IFEval](https://arxiv.org/abs/2311.07911), [ARC](https://arxiv.org/abs/1803.05457), "
              "[Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B), [TRL GKD](https://huggingface.co/docs/trl/gkd_trainer), "
              "[IFEval implementation](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.2/lm_eval/tasks/ifeval).", ""]
    Path("results/report.md").write_text("\n".join(line.rstrip() for line in "\n".join(lines).splitlines()) + "\n")
    artifacts = [*Path("data").glob("*.jsonl"), *Path("results").glob("*/ifeval.jsonl"),
                 *Path("results").glob("*/arc.jsonl"), *Path("runs").glob("E*/trajectories.jsonl")]
    write_json("results/artifact_hashes.json", {str(p): digest_file(p) for p in sorted(artifacts)})
    print(Path("results/report.md").resolve())


if __name__ == "__main__":
    main()
