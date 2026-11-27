## Long-completion results

The original student outscored every trained checkpoint on strict IFEval prompt accuracy, strict instruction accuracy and ARC-Challenge. All 12 paired 95% intervals for the changes from baseline fell below zero.
All four methods reduced held-out loss on teacher responses. This measures how closely a student predicts the teacher on that validation set.

**Hard and soft labels.** On exactly the same 4,728,146 fixed response tokens, replacing hard labels with forward KL changed strict prompt accuracy by -1.11 percentage points, strict instruction accuracy by -0.72 points, and ARC by -1.00 points.

**Student-generated training contexts.** Moving from fully off-policy to fully on-policy KL changed strict instruction accuracy by +2.16 points and ARC by -2.00 points. On-policy KL consumed 58.06 allocated training GPU-hours compared with 1.54 for off-policy KL. The shared teacher-data generation cost is reported separately.

**Hybrid training.** The hybrid used fresh student answers for 2,568 of 5,000 prompts. Its strict instruction accuracy changed by +2.28 points relative to off-policy KL and +0.12 relative to on-policy KL. The corresponding ARC differences were +0.75 and +2.75 points. Across the four direct comparisons between training methods, 12 of 12 paired 95% intervals included zero.

**Response length.** With a 16,384-token ceiling, 10 of 5,000 fixed teacher answers reached the limit. In on-policy training, 156 answers reached it and contributed 35.3% of response tokens. On IFEval, the original student reached the ceiling on 23 prompts. Counts for SFT, off-policy, hybrid and on-policy training were 128, 121, 87 and 72. Prompt budgets and optimizer steps were matched. Changing the training contexts also changed the token budget and compute.

Each condition was trained once. Unadjusted paired intervals resample evaluation items and quantify uncertainty for these checkpoints. Variation across training seeds and the effect of increasing the ceiling past the studied value remain open.
