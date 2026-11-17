# Research Proposal

## Off-Policy, On-Policy, and Hybrid Knowledge Distillation for Small Language Models

### 1. Abstract

Knowledge distillation (KD) is a technique for transferring knowledge from a large teacher model to a smaller student model. Traditional language model distillation typically relies on fixed teacher-generated responses, while on-policy distillation allows the student to learn from its own generated sequences.

This project investigates how different knowledge distillation strategies affect the performance of a small general-purpose language model.

We compare hard-label supervised fine-tuning (SFT), soft-label off-policy distillation, on-policy distillation, and hybrid distillation with different mixing ratios.

All experiments use full-parameter fine-tuning and share the same initial student checkpoint. Performance is evaluated on instruction following and general reasoning tasks.

The goal is to determine whether incorporating student-generated sequences improves knowledge transfer and whether a hybrid strategy can outperform purely off-policy or on-policy training.

### 2. Research Questions

This project focuses on three research questions:

**RQ1: Hard Labels vs. Soft Labels**

Does soft-label knowledge distillation outperform traditional hard-label supervised fine-tuning?

**RQ2: Off-Policy vs. On-Policy Distillation**

How does the proportion of student-generated training sequences affect the performance of a distilled language model?

**RQ3: Hybrid Knowledge Distillation**

Can combining off-policy and on-policy distillation achieve better performance than using either strategy independently?

### 3. Models and Hardware

We use two pretrained, instruction-tuned models from the Qwen3 family.

| Component | Model |
| --- | --- |
| Teacher Model | Qwen3-4B |
| Student Model | Qwen3-0.6B |
| Hardware | 4 × NVIDIA RTX 3090 (24GB each) |
| Training Strategy | Full-Parameter Fine-Tuning |
| Framework | PyTorch, Hugging Face Transformers, TRL |

The teacher model remains frozen during distillation. The student model is initialized from the same original Qwen3-0.6B checkpoint for every experiment.

Both models operate in non-thinking mode to reduce training costs and maintain a consistent generation format.

No additional pre-training, reinforcement learning, reward models, or human preference annotations are required.

### 4. Training Dataset

We use the **UltraChat 200k** dataset as the source of general-purpose instruction prompts.

To keep the project computationally manageable, we sample approximately **5,000 prompts** covering a variety of tasks, including:

- General question answering
- Instruction following
- Summarization and rewriting
- Commonsense reasoning
- Creative writing and text generation

The teacher model generates a response for each selected prompt. These responses form a fixed dataset for hard-label SFT and off-policy knowledge distillation.

All experiments use the same training prompts. The fixed teacher responses are generated once and reused across experiments.

We also reserve a small validation set that does not overlap with the training set.

### 5. Methodology

We conduct five experiments, including one untrained baseline.

| ID | Training Method | λ | β |
| --- | --- | --- | --- |
| E0 | Original Student (Baseline) | — | — |
| E1 | Hard-Label Off-Policy SFT | — | — |
| E2 | Soft-Label Off-Policy GKD | 0 | 0 |
| E3 | Hybrid GKD | 0.5 | 0 |
| E4 | On-Policy GKD | 1 | 0 |

All trainable experiments (E1–E4) perform full-parameter fine-tuning, without LoRA or other parameter-efficient adapters.

**Experiment E0: Original Student Baseline**

Evaluate the original Qwen3-0.6B model without additional training. This establishes the baseline performance of the student.

**Experiment E1: Hard-Label Off-Policy Distillation**

Fine-tune the student using teacher-generated responses as hard labels.

The training objective is standard cross-entropy loss:

\[
\mathcal{L}_{SFT}
= -\sum_t \log P_S(y_t\mid x,y_{<t})
\]

**Experiment E2: Soft-Label Off-Policy Distillation**

Train the student using the teacher's token-level probability distributions on the same fixed response sequences used in E1.

The training objective uses forward KL divergence:

\[
\mathcal{L}_{KD}
=D_{KL}(P_T\parallel P_S)
\]

This experiment isolates the effect of soft-label supervision compared with hard-label SFT.

**Experiment E3: Hybrid Knowledge Distillation**

Apply Generalized Knowledge Distillation (GKD) with λ = 0.5.

Training alternates probabilistically between two sources:

- Fixed teacher-generated sequences (off-policy).
- Student-generated sequences (on-policy).

The teacher provides token-level probability distributions for both types of sequences.

**Experiment E4: On-Policy Knowledge Distillation**

Apply GKD with λ = 1.0.

The student generates its own responses, and the teacher provides token-level supervision on the student-generated contexts.

This experiment examines the effectiveness of knowledge distillation using exclusively on-policy training sequences.

For E2–E4, the divergence parameter β is fixed at 0, corresponding to forward KL divergence.

### 6. Training Configuration

The initial training configuration is:

| Parameter | Configuration |
| --- | --- |
| Training Prompts | 5,000 |
| Fine-Tuning | Full-Parameter |
| Training Epochs | 1 |
| GKD λ | 0, 0.5, 1 |
| GKD β | 0 (Forward KL) |
| Precision | BF16 |
| Maximum Completion Length | 128–256 tokens |
| Sequence Length | 512–1024 tokens |
| Gradient Checkpointing | Enabled |

Exact batch sizes, learning rates, and sequence lengths will be finalized after a small pilot run.

To ensure a fair comparison, all experiments use the same initial student weights, training prompts, and evaluation configuration.

We aim to keep optimizer steps and effective training-token budgets comparable. Because on-policy generation introduces additional computational overhead, GPU time and token counts will also be recorded.

### 7. Evaluation

We evaluate the teacher, the original student, and all four fine-tuned student models using two benchmarks.

**Benchmark 1: IFEval**

IFEval measures a language model's ability to follow explicit instructions.

Primary metrics:

- Strict Instruction Accuracy
- Strict Prompt Accuracy

We use the complete benchmark.

**Benchmark 2: ARC-Challenge**

ARC-Challenge measures general knowledge and reasoning through challenging multiple-choice questions.

Primary metric:

- Multiple-choice Accuracy

To reduce evaluation costs, we use a fixed subset of approximately 400 examples, shared across all models.

Both benchmarks are evaluated using consistent prompting and scoring protocols.

IFEval serves as the primary metric, while ARC-Challenge provides a secondary measure of general knowledge and reasoning retention.

### 8. Experimental Analysis

The main comparisons are:

| Comparison | Research Objective |
| --- | --- |
| E0 vs. E1–E4 | Effectiveness of additional knowledge distillation |
| E1 vs. E2 | Hard-label vs. soft-label supervision |
| E2 vs. E4 | Off-policy vs. on-policy training |
| E2 vs. E3 vs. E4 | Effect of the on-policy mixing ratio |

We will report results using the following table:

| Model | IFEval ↑ | ARC-Challenge ↑ |
| --- | --- | --- |
| Teacher: Qwen3-4B | TBD | TBD |
| E0: Original Student | TBD | TBD |
| E1: Hard-Label SFT | TBD | TBD |
| E2: GKD (λ=0) | TBD | TBD |
| E3: GKD (λ=0.5) | TBD | TBD |
| E4: GKD (λ=1) | TBD | TBD |

Additional analysis will include:

- Performance changes relative to the original student.
- The relationship between λ and benchmark performance.
- Training loss and convergence behavior.
- GPU time and computational costs.
- Selected examples of successful and unsuccessful generations.

### 9. Expected Outcomes and Hypotheses

We formulate the following hypotheses:

**H1:** Soft-label distillation may outperform hard-label SFT because teacher probability distributions provide richer supervision than individual target tokens.

**H2:** On-policy distillation may improve performance by exposing the student to its own generated contexts, reducing the mismatch between training and inference distributions.

**H3:** Hybrid distillation may achieve a favorable balance between high-quality fixed teacher responses and student-generated training sequences.

These are hypotheses rather than assumed results. The experiments may also demonstrate that simpler off-policy approaches are more effective under limited training budgets.

### 10. Project Scope and Limitations

This project intentionally excludes training a model from scratch, reinforcement learning, reward modeling, and large-scale hyperparameter searches.

Several limitations should be acknowledged:

1. The student is already instruction-tuned, so this project evaluates additional knowledge transfer rather than distillation into an untrained language model.
2. The relatively small training dataset may limit performance improvements.
3. Different on-policy ratios produce different training sequences and computational costs.
4. Two evaluation benchmarks cannot capture every aspect of general-purpose language model performance.
5. Results from a small number of training runs may be sensitive to random seeds and optimization settings.

Despite these limitations, the proposed experiments provide a controlled comparison of several important knowledge distillation strategies within a practical computational budget.

### 11. Expected Deliverables

The final project will include:

1. Reproducible training and evaluation code.
2. A fixed teacher-generated instruction dataset.
3. Four fine-tuned student checkpoints.
4. Benchmark results comparing all experimental conditions.
5. Performance and training-efficiency visualizations.
6. A final report discussing methodology, results, limitations, and conclusions.

### 12. Conclusion

This project systematically compares hard-label supervised fine-tuning, soft-label off-policy distillation, hybrid distillation, and on-policy distillation for compressing general-purpose language model capabilities into a smaller student.

By varying the source of training sequences while maintaining a consistent teacher, student, and evaluation framework, the project aims to provide empirical evidence on the effectiveness of different knowledge distillation strategies.

The proposed methodology balances research depth, experimental clarity, and computational feasibility, making it suitable for a university-level final project.
