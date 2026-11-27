# Long-completion Qwen3 distillation

This study compares the original Qwen3-0.6B student, hard-label SFT and three full-vocabulary forward-KL conditions. The frozen teacher is Qwen3-4B. The training and IFEval completion ceiling is 16,384 tokens. EOS ends each individual response. The shared training set has 5,000 prompts, and validation uses 200 separate prompts.

The original student scored above all four trained checkpoints on both strict IFEval metrics and the 400-question ARC subset. All 12 paired intervals for the direct comparisons among training methods included zero. The [final report](results/report.md), [score table](results/summary.csv) and [paired intervals](results/comparisons.json) give the results, training costs and completion lengths.

The [protocol](docs/protocol.md) describes the common training and evaluation settings. The [configuration](config.json) specifies the active token limits. The study uses physical GPUs 0–3 on the research server. Training records and full student checkpoints live in `runs/E1` through `runs/E4`. The compact metrics and figures are generated in `results/`.

For an independent rerun, use a new checkout at the frozen experiment revision `2160b8f`. This revision predates the archived scores. Create `.venv` with the recorded dependencies and make the model snapshots in `config.json` available. From the checkout root, run `.venv/bin/python -m kd.orchestrate --phase prepare` to regenerate the fixed prompts and benchmark inputs, then `bash studies/long/run.sh` for generation, training, evaluation and reporting. The launcher saves its phase schedules in `logs/`. To resume an interrupted run, invoke it in the same run directory.
