#!/usr/bin/env bash
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
run="$repo/reproductions/completion-1024"
mkdir -p "$run/data" "$run/logs"
exec 9>"$run/logs/controller.lock"
flock -n 9
src="$repo/studies/completion-1024/config.json"
if [[ -e "$run/config.json" ]]; then cmp "$src" "$run/config.json"; else cp "$src" "$run/config.json"; fi
for name in train_prompts.jsonl validation_prompts.jsonl arc.jsonl ifeval.jsonl manifest.json; do
    if [[ -e "$run/data/$name" ]]; then cmp "data/$name" "$run/data/$name"; else cp "data/$name" "$run/data/$name"; fi
done
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH="$repo"
cd "$run"
python="$repo/.venv/bin/python"
if [[ ! -f data/audit.json ]]; then
    "$python" -m kd.orchestrate --phase generate --wait
    cp logs/status.json logs/generation_schedule.json
fi
if [[ ! -f results/job_schedule.json ]]; then
    "$python" -m kd.orchestrate --phase research --wait
    cp logs/status.json results/job_schedule.json
fi
CUDA_VISIBLE_DEVICES= "$python" -m kd.report
CUDA_VISIBLE_DEVICES= "$python" "$repo/studies/completion-1024/compare.py"
