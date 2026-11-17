# Off-policy, on-policy, and hybrid Qwen3 distillation

Controlled comparison of the original Qwen3-0.6B (E0), full-parameter hard-label
SFT (E1), and exact forward-KL distillation with student rollout fractions 0,
0.5, and 1 (E2–E4). The frozen teacher is the original Qwen3-4B.

All models use the non-thinking chat template. Training uses the same 5,000
unique UltraChat prompts, a separate 200-prompt validation set, and one epoch.
E1/E2 share the exact teacher-generated token sequences. Soft targets cover the
entire vocabulary at temperature 1. Completion-only losses include genuine EOS
tokens and exclude prompt/padding tokens. Losses are normalized over all response
tokens in each effective batch, including the final partial batch.

## Results and artifacts

All four fine-tuned students scored below the original student on both benchmarks
under this short-completion, single-seed protocol. See the report for paired
uncertainty estimates and the substantial training-response truncation limitation.

The completed results are in [the report](results/report.md),
[the metric table](results/summary.csv), and [paired confidence intervals](results/comparisons.json).
The fixed dataset and four full student checkpoints are retained both in this
workspace and on the GPU server; they are excluded from Git because of their size.
Compact training traces and provenance are archived in `results/training/`.

## Run

The research server checkout is `/home/data2/xq/projects/LLM-distillation` on
`xq@10.123.4.22`. Only physical GPUs 0–3 are allocated; 4–7 are never selected.
The orchestrator enforces this allocation and uses at most four GPUs concurrently.

```bash
# Recorded environment: Python 3.10.16, PyTorch 2.5.1+cu124.
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
# Set local model paths in config.json; dataset commits are pinned.
# An empty output directory prevents reuse of the archived benchmark results.
task_root="$PWD"
task_python="$task_root/.venv/bin/python"
mkdir -p reproductions
mkdir reproductions/new-run
cp config.json dataset_revisions.json reproductions/new-run/
cd reproductions/new-run
export PYTHONPATH="$task_root"
"$task_python" -m kd.orchestrate --phase prepare
"$task_python" -m kd.orchestrate --phase generate
"$task_python" -m kd.orchestrate --phase pilot
# To reproduce this experiment, retain the recorded final configuration.
nohup "$task_python" -m kd.orchestrate --phase research > logs/research.log 2>&1 &
```

Download NLTK `punkt`, `punkt_tab`, and `averaged_perceptron_tagger_eng` to
`~/nltk_data` before evaluation (NLTK requires a private download directory). The server environment reuses the existing
PyTorch installation through a `.pth` entry while installing additional packages
in this project's own virtual environment. `environment.freeze.txt` records the
resolved versions. No packages are installed into another project's environment.

The commands above create a separate run. To resume an interrupted run, use its
existing directory and rerun only the research phase with the same configuration.
A fresh Git checkout still contains archived compact results, so do not launch a
new experiment in its root. Do not rerun preparation over the archived dataset. The original preparation
manifest is tracked at `results/data_manifest.json`; its configuration predates
the pilot-selected batch sizes. The final training configuration is `config.json`
and is duplicated in every run's metadata.

Training checkpoints and resumable optimizer state are stored in `runs/E*/`.
Rerunning the research orchestrator skips completed jobs and resumes interrupted
training at the latest 50-step checkpoint. Generation and evaluation persist each
completed batch. Inspect `logs/status.json` and per-job log files for progress.

## Protocol and outputs

- 512-token maximum prompts, 256-token maximum training completions; long prompts
  are excluded before selecting the sample, rather than silently truncated.
- FP32 master weights/AdamW states with BF16 autocast; gradient checkpointing;
  full-parameter updates. Output projections are checkpointed in chunks to compute
  exact full-vocabulary forward KL without approximating the distribution.
- Teacher/student generation: temperature 0.7, top-p 0.8, top-k 20.
- All 541 IFEval prompts: greedy non-thinking generation, up to 2,048 tokens;
  strict prompt and micro-averaged instruction accuracy from lm-evaluation-harness.
- ARC-Challenge: fixed seeded 400-example test subset, zero-shot chat, select the
  answer letter with maximum next-token likelihood. This is a custom, consistently
  applied chat protocol, not the harness's default continuation scoring.
- Dataset revisions, exact rows, token IDs, model/config hashes: `data/manifest.json`.
- Per-example benchmark responses/scores: `results/<model>/`; training curves,
  counts, peak memory and allocated GPU hours: `runs/<experiment>/`.

The comparison controls examples and optimizer steps. Actual response-token counts
can differ for on-policy conditions and are reported, rather than assumed equal.
One seed per condition cannot establish robustness across training seeds. Paired
bootstrap intervals quantify evaluation-set uncertainty only.

Implementation references: [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B),
[TRL GKD](https://huggingface.co/docs/trl/gkd_trainer),
[IFEval scoring](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.2/lm_eval/tasks/ifeval).

## Recheck and regenerate analysis

With the recorded data and checkpoints present on the server:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m kd.audit
CUDA_VISIBLE_DEVICES= .venv/bin/python -m kd.report
```

The full audit verifies original model and prepared-data hashes, training
trajectory coverage and fixed-response reuse, benchmark IDs and score aggregation,
and exported checkpoint hashes. The report includes `docs/conclusions.md` when
present. `results/job_schedule.json` preserves the completed schedule; training
and evaluation outputs record the frozen model-run Git revision. Later analysis
and documentation changes do not change those recorded results.
