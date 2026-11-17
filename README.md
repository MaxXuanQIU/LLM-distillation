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

## Run

The research server checkout is `/home/data2/xq/projects/LLM-distillation` on
`xq@10.123.4.22`. Only physical GPUs 0–3 are allocated; 4–7 are never selected.
The orchestrator enforces this allocation and uses at most four GPUs concurrently.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Set the local teacher/student paths in config.json.
.venv/bin/python -m pytest -q
.venv/bin/python -m kd.orchestrate --phase prepare
.venv/bin/python -m kd.orchestrate --phase generate
.venv/bin/python -m kd.orchestrate --phase pilot
# Review pilot memory/throughput, freeze config.json, then:
nohup .venv/bin/python -m kd.orchestrate --phase research > logs/research.log 2>&1 &
```

Download NLTK `punkt`, `punkt_tab`, and `averaged_perceptron_tagger_eng` to
`.cache/nltk_data` before evaluation. The server environment reuses the existing
PyTorch installation through a `.pth` entry while installing additional packages
in this project's own virtual environment. `environment.freeze.txt` records the
resolved versions. No packages are installed into another project's environment.

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
