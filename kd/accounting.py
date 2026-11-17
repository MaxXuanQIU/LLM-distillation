"""Summarize completed-run costs, output lengths, and the four-GPU schedule."""
import json
from pathlib import Path

import numpy as np

from kd.common import write_json


def summarize(metrics, train, histories, ifeval, baseline_val, generation):
    lengths = {}
    for name, records in ifeval.items():
        values = np.array([r["completion_tokens"] for r in records])
        lengths[name] = {"mean_tokens": float(values.mean()), "median_tokens": float(np.median(values)),
                         "p95_tokens": float(np.quantile(values, .95)),
                         "capped": sum(r["truncated"] for r in records)}
    timing = {
        "teacher_generation_gpu_hours": sum(r["seconds"] for r in generation)/3600,
        "teacher_generation_tokens": sum(r["new_tokens"] for r in generation),
        "training_loop_gpu_hours": sum(t["allocated_gpu_hours"] for t in train.values()),
        "evaluation_loop_gpu_hours": sum(m["evaluation_seconds_this_session"] for m in metrics.values())/3600,
        "baseline_validation_gpu_hours": baseline_val["seconds"]/3600,
    }
    status_path = Path("logs/status.json")
    if status_path.exists():
        state = json.loads(status_path.read_text())
        if state.get("finished") and state.get("phase") == "research":
            timing["main_phase_wall_hours"] = (state["finished"]-state["started"])/3600
            timing["main_phase_job_gpu_hours"] = sum((j["finished"]-j["started"])*len(j["gpus"])/3600
                                                    for j in state["jobs"].values())
            events = []
            for name, job in state["jobs"].items():
                assert job["status"] == "complete"
                for gpu in job["gpus"]:
                    assert gpu in {"0", "1", "2", "3"}
                    events.extend([(job["started"], 1, gpu, name), (job["finished"], -1, gpu, name)])
            active, peak = set(), 0
            for _, change, gpu, name in sorted(events):
                if change == 1:
                    assert gpu not in active, f"Overlapping GPU assignment: {gpu}, {name}"
                    active.add(gpu)
                else:
                    active.remove(gpu)
                peak = max(peak, len(active))
            assert peak <= 4 and not active
            timing["peak_concurrent_assigned_gpus"] = peak
            write_json("results/job_schedule.json", state)
    write_json("results/run_summary.json", {"timing": timing, "ifeval_lengths": lengths,
               "original_student_validation_ce": baseline_val["validation_cross_entropy"]})
    for name in train:
        record = {"summary": train[name], "history": histories[name],
                  "metadata": json.loads(Path(f"runs/{name}/metadata.json").read_text())}
        weight_check = Path(f"runs/{name}/weight_update_check.json")
        if weight_check.exists():
            record["weight_update_check"] = json.loads(weight_check.read_text())
        write_json(f"results/training/{name}.json", record)
    diagnostics = {}
    for name, history in histories.items():
        t = train[name]
        with Path(f"runs/{name}/trajectories.jsonl").open() as f:
            capped = sum(json.loads(line)["truncated"] for line in f if line.strip())
        diagnostics[name] = {
            "first_20_steps_mean_loss": float(np.mean([r["loss"] for r in history[:20]])),
            "last_20_steps_mean_loss": float(np.mean([r["loss"] for r in history[-20:]])),
            "max_preclip_gradient_norm": max(r["grad_norm"] for r in history),
            "training_capped_responses": capped,
            "examples": t["examples"],
            "generation_fraction_of_training_time": t["generation_seconds"]/t["training_seconds"],
            "training_wall_minutes": t["training_seconds"]/60,
        }
    write_json("results/training_diagnostics.json", diagnostics)
    baseline = {r["id"]: r for r in ifeval["E0"]}
    behavior = {}
    for name in train:
        improved = regressed = capped_regressions = 0
        for row in ifeval[name]:
            before = baseline[row["id"]]["prompt_level_strict_acc"]
            after = row["prompt_level_strict_acc"]
            improved += not before and after
            regressed += before and not after
            capped_regressions += before and not after and row["truncated"]
        behavior[name] = {"baseline_fail_to_pass": improved, "baseline_pass_to_fail": regressed,
                          "regressions_reaching_evaluation_cap": capped_regressions}
    write_json("results/behavior_diagnostics.json", behavior)
    return timing, lengths
