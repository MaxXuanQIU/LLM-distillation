# Protocol decisions

The attached proposal is preserved in `research_proposal.md`. The experiment uses
its five student conditions and evaluates the teacher as a sixth model.

## Controls

All conditions start from the same original instruction-tuned Qwen3-0.6B weights,
use one fixed 5,000-prompt training set, one epoch, and the same AdamW settings.
E1 and E2 use exactly the same stored teacher response token IDs. E2–E4 use full
vocabulary forward KL, with no top-k approximation or sampled soft targets.
Teacher and student tokenizers are checked for identical vocabularies and chat
prefixes. The teacher remains frozen. Every student parameter is trainable.

The custom PyTorch training loop implements TRL's beta=0 GKD objective. A numerical
test checks equality with TRL 0.24.0, and separate tests compare the chunked loss
and every parameter gradient with a dense reference model. Chunking changes peak
memory, not the loss definition. Prompt and padding targets are masked; a real
EOS is included. A length-capped response is not given an artificial EOS.

BF16 autocast is used with FP32 master parameters and optimizer states. This
avoids losing small AdamW updates in BF16 parameter rounding. The teacher resides
on a second GPU for each soft-label run. The maximum global allocation is physical
GPUs 0–3; GPUs 4–7 are not selected by any job.

## Sampling and budgets

The seed is 20261117. UltraChat train_sft rows are randomly permuted before selecting
unique first user turns that fit the 512-token prompt budget. Exact normalized
matches to IFEval prompts and ARC questions are excluded. The first 5,000 selected
rows are training; the next 200 are validation. This is a length-filtered sample,
not an unqualified random sample of all conversations.

Teacher and student rollouts sample at temperature 0.7, top-p 0.8, top-k 20 in
non-thinking mode. The training completion cap is the proposal's upper bound,
256 tokens. Initial teacher samples show that many responses reach this cap;
the final report must quantify this limitation instead of treating all examples
as complete answers. The same generation cap applies to every training condition.

The pilot tests memory and numerical stability. Batch size can be finalized from
the pilot; evaluation benchmarks are not used for tuning. The final configuration
is stored both at the repository root and in each run's metadata.

All runs see the same prompt order. GKD source choices use an independent seeded
Bernoulli stream per microbatch. Effective batches are normalized by their actual
supervised token count, including the final partial batch. Training trajectories
record prompt IDs, source choices, and exact completion token IDs. Equal prompt
and step budgets do not imply equal response-token counts; both are reported.

## Evaluation and inference

IFEval uses all 541 prompts, the non-thinking chat template, greedy decoding, and
a 2,048-token generation cap. The pinned lm-evaluation-harness scorer supplies
strict prompt and instruction metrics. Instruction accuracy is micro-averaged,
not an average of prompt-level instruction fractions. Language detection is seeded.

ARC uses a fixed seeded 400-example subset of the test split. A zero-shot user
message gives the question and choices and requests only the answer letter. The
predicted answer maximizes the next-token likelihood among candidate letters.
The same protocol is used for all six models. It is not the default harness ARC
continuation-scoring protocol, so external leaderboard comparisons are inappropriate.

Ten thousand paired bootstrap replicates resample whole IFEval prompts or ARC
questions. Instruction intervals preserve clustering within prompts. Pairwise
intervals are exploratory and not corrected for multiple comparisons. They do
not capture variation across training seeds. No hypothesis is assumed true.

## Artifacts and timing

The server stores the dataset and four full student checkpoints. Git records code
on `main`. Dataset revisions, model file hashes, package versions, exact prompts,
training trajectories, and per-example evaluation outputs make the run auditable.
The final artifact audit checks all 5,000 unique training IDs, token counts,
off-policy sequence equality, optimizer steps, benchmark sizes, and checkpoint hashes.

Allocated GPU hours are training wall time multiplied by the number of GPUs assigned
to that run. They include generation and idle time; they are not energy or integrated
GPU utilization. Generation and evaluation timings are recorded separately. Large
artifacts stay outside Git; compact metrics, charts, and the report are retained.
