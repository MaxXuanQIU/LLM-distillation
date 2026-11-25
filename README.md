# Long-completion Qwen3 distillation

This study compares the original Qwen3-0.6B student, hard-label SFT and three full-vocabulary forward-KL conditions. The frozen teacher is Qwen3-4B. The training and IFEval completion ceiling is 16,384 tokens. EOS ends each individual response. The shared training set has 5,000 prompts, and validation uses 200 separate prompts.

The [protocol](docs/protocol.md) describes the common training and evaluation settings. The [configuration](config.json) specifies the active token limits. The study uses physical GPUs 0–3 on the research server. Training records and full student checkpoints live in `runs/E1` through `runs/E4`. The compact metrics and figures are generated in `results/`.

From the checkout root on the server, use the project's Python environment with dependencies from `requirements.txt`. The launcher reads the prepared inputs in `data/`. To regenerate these inputs in a fresh checkout, first run `.venv/bin/python -m kd.orchestrate --phase prepare`. Start the study with `bash studies/long/run.sh`. The launcher saves its phase schedules in `logs/`. Use the existing directory to resume incomplete model jobs.
