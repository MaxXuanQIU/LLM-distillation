#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
mkdir -p logs
exec 9>logs/long-study.lock
flock -n 9
export CUDA_VISIBLE_DEVICES=0,1,2,3
python=.venv/bin/python
if [[ ! -f data/audit.json ]]; then
    "$python" -m kd.orchestrate --phase generate --wait
    cp logs/status.json logs/generation_schedule.json
fi
if [[ ! -f logs/research_schedule.json ]]; then
    "$python" -m kd.orchestrate --phase research --wait
    cp logs/status.json logs/research_schedule.json
fi
CUDA_VISIBLE_DEVICES= "$python" -m kd.report
