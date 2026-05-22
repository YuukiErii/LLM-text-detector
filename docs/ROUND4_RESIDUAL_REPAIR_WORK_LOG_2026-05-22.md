# Round4 Residual Repair Work Log

Updated: 2026-05-22

This note records the first Round4 optimization checkpoint after the completed
three-round review in:

```text
docs/THREE_ROUND_OPTIMIZATION_REVIEW_AND_95_ROUTE_2026-05-22.md
```

Round4 follows the strict route:

```text
paired residual data -> weighted DeBERTa retrain -> human-style guard ->
precision-guarded local override -> final teacher-test diagnostic
```

Teacher-test labels remain out of training, threshold selection, routing, guard
tuning, and model selection. Teacher-test text is used only for exact-text
leakage exclusion when building new train/dev data.

## 1. Step 1 Completed: Residual Dataset Rebuild

New script:

```text
src/data/build_round4_residual_dataset.py
```

Main command:

```powershell
.\.venv\Scripts\python.exe src\data\build_round4_residual_dataset.py
```

The script creates:

```text
data/processed/round4_hard_human_mirror_seed.jsonl
data/processed/round4_hard_llm_positive_seed.jsonl
data/processed/round4_residual_train.jsonl
data/processed/round4_residual_dev_hardpos.jsonl
data/processed/round4_residual_dev_hardneg.jsonl
data/processed/round4_residual_spotcheck.jsonl
data/processed/round4_residual_report.json
```

Additional old-prose human mirror source created from local Gutenberg fiction:

```text
data/processed/round4_old_prose_human_mirror_candidates.jsonl
data/processed/round4_old_prose_human_mirror_candidates_report.json
```

Old-prose candidate command:

```powershell
.\.venv\Scripts\python.exe src\data\build_hard_negative_human_seed.py `
  --existing_human data\processed\human_seed_combined_with_hardneg.jsonl data\processed\round4_hard_human_mirror_seed.jsonl data\processed\lit_academic_poetry_valid.jsonl data\processed\lit_academic_poetry_internal_test.jsonl `
  --output data\processed\round4_old_prose_human_mirror_candidates.jsonl `
  --report data\processed\round4_old_prose_human_mirror_candidates_report.json `
  --poetry_target 0 `
  --literature_target 1200 `
  --academic_target 0 `
  --seed 20260522
```

## 2. Step 1 Acceptance

Current acceptance result:

| Check | Result |
| --- | ---: |
| hard human negatives >= 3000 | pass |
| hard LLM positives >= 3000 | pass |
| old-prose human mirrors >= 800 | pass |
| poetry/freeverse human mirrors >= 1000 | pass |
| manual spot check >= 100 rows | pass |
| no teacher-test exact text duplicates | pass |
| hard human : hard LLM ratio >= 1 : 1 | pass |

Key counts from `data/processed/round4_residual_report.json`:

| Item | Count |
| --- | ---: |
| hard human mirrors | 4727 |
| hard LLM positives | 3674 |
| hard human : hard LLM ratio | 1.2866 |
| old-prose human mirrors | 800 |
| poetry/freeverse human mirrors | 1500 |
| natural academic human mirrors | 1000 |
| old-prose LLM positives | 300 |
| hard-positive dev rows | 500 |
| hard-negative dev rows | 500 |
| teacher-test exact duplicates | 0 |
| round4 train rows | 17434 |

Round4 residual bucket coverage:

| Round4 bucket | Human mirrors | LLM positives |
| --- | ---: | ---: |
| academic_formal | 1000 | 672 |
| general_prose | 1000 | 800 |
| literary_old_prose | 800 | 300 |
| literary_short_fragment | 427 | 699 |
| poetry_classical | 500 | 303 |
| poetry_freeverse | 1000 | 900 |

Important implementation detail:

```text
The old `bucket` field still stores the text-feature bucket from
assign_text_bucket.py. Round4 adds `round4_bucket` so source-known old-prose
mirrors and old-fiction rewrite prompts are not lost just because the older
heuristic bucketter labels them as general_prose or literary_short_fragment.
```

This matches the Round4 thesis: the model must learn source-vs-style evidence
inside a high-style region, not only global text-shape buckets.

## 3. Verification

Syntax check:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\data\build_round4_residual_dataset.py
```

Result:

```text
passed
```

## 7. Step 2 Result: Weighted DeBERTa Retrain

Training completed:

```text
outputs/models/round4_deberta_weighted_residual/
```

Best checkpoint:

```text
outputs/models/round4_deberta_weighted_residual/checkpoint-2180
```

Final saved model:

```text
outputs/models/round4_deberta_weighted_residual/best_model
```

Validation history:

| Epoch | Validation F1 | Precision | Recall | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.9535 | 0.9496 | 0.9574 | 0.9903 |
| 2 | 0.9365 | 0.8928 | 0.9846 | 0.9929 |
| 3 | 0.9493 | 0.9164 | 0.9846 | 0.9937 |

Best-checkpoint metrics at threshold `0.5`:

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.9543 | 0.9496 | 0.9574 | 0.9535 | 43 | 36 |
| internal_test | 0.9445 | 0.9321 | 0.9563 | 0.9441 | 59 | 37 |
| hardpos dev | 0.5660 | 1.0000 | 0.5660 | 0.7229 | 0 | 217 |

Round4 DeBERTa is not a global replacement for Step7:

| Split | Step7 F1 | Round4 DeBERTa F1 | Main Change |
| --- | ---: | ---: | --- |
| internal_test | 0.9564 | 0.9441 | worse global balance |
| hardpos dev | 0.5876 | 0.7229 | better residual recall |
| hardneg dev | N/A | N/A | FP rises from 26 to 53 |

Interpretation:

```text
Round4 DeBERTa learned a useful hard-positive signal, but it over-shifts toward
LLM and creates too many human false positives. It can only be considered as a
local override signal, not as a final classifier.
```

## 8. Step 4 Result: Residual Override Gate

Command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\tune_round4_residual_override.py `
  --tune_set internal_test step7=outputs\predictions\round4_step7_internal_test_predictions.jsonl round4=outputs\predictions\round4_deberta_internal_test_predictions.jsonl guard=outputs\predictions\round4_human_style_guard_internal_test_predictions.jsonl `
  --tune_set hardpos step7=outputs\predictions\round4_step7_hardpos_predictions.jsonl round4=outputs\predictions\round4_deberta_hardpos_predictions.jsonl guard=outputs\predictions\round4_human_style_guard_hardpos_predictions.jsonl `
  --tune_set hardneg step7=outputs\predictions\round4_step7_hardneg_predictions.jsonl round4=outputs\predictions\round4_deberta_hardneg_predictions.jsonl guard=outputs\predictions\round4_human_style_guard_hardneg_predictions.jsonl
```

Output:

```text
outputs/models/round4_residual_override/rules.json
outputs/models/round4_residual_override/residual_override_tuning_report.json
outputs/evaluation/round4_residual_override_tuning_report.md
```

Result:

| Item | Value |
| --- | ---: |
| aligned non-teacher rows | 2731 |
| candidate rules searched | 3601 |
| feasible rules | 1 |
| feasible non-empty override rules | 0 |

The only feasible rule is the disabled/no-op baseline:

| Split | F1 | FP | FN | Overrides | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 27 | 46 | 0 | 0 | 0 |
| hardpos | 0.5876 | 0 | 292 | 0 | 0 | 0 |
| hardneg | 0.0000 | 26 | 0 | 0 | 0 | 0 |

Decision:

```text
Do not advance Round4 v1 to teacher-test diagnostic.
It improves hard-positive recall only by violating hard-negative/internal FP
constraints. The strict route therefore keeps Step7 as the current final model.
```

## 9. Next Recommended Round4 Move

The next attempt should not simply lower override thresholds. That failure mode
is already visible: it repairs hard positives by breaking hard humans.

Next data/model actions:

1. Add more matched old-prose LLM positives from non-ChatGPT generators. The
   current Round4 old-prose LLM bucket reached the minimum `300`, but it is
   still much smaller than poetry/freeverse and general-prose coverage.
2. Add a second human-style guard trained specifically on the hardneg rows that
   Round4 DeBERTa flips. The current guard protects only 21.4% of hard-human
   dev at the conservative threshold.
3. Try a lower-learning-rate or one-epoch DeBERTa variant. Epoch 1 was best;
   later epochs became recall-heavy and precision-weak.
4. Consider threshold calibration for Round4 DeBERTa on non-teacher splits, but
   only as an input to the override gate, not as a global replacement.

## 6. Step 4 Prepared: Residual Override Tuning

New script:

```text
src/evaluation/tune_round4_residual_override.py
```

Purpose:

```text
default = Step7 prediction
allow only Step7 human -> LLM overrides
require high Round4 probability, sufficient margin over Step7, and no
human-style guard veto
```

The script searches rules only on non-teacher splits. It is intended to run
after Round4 DeBERTa predictions are available for:

```text
internal_test
round4_residual_dev_hardpos
round4_residual_dev_hardneg
```

Expected command shape:

```powershell
.\.venv\Scripts\python.exe src\evaluation\tune_round4_residual_override.py `
  --tune_set internal_test step7=... round4=... guard=... `
  --tune_set hardpos step7=... round4=... guard=... `
  --tune_set hardneg step7=... round4=... guard=...
```

Selection constraints:

```text
hardneg false positives must not exceed Step7
internal-test F1 must not regress beyond tolerance
hardpos recall must improve or stay no worse than Step7
```

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\evaluation\tune_round4_residual_override.py
```

Result:

```text
passed
```

## 4. Step 2 Candidate Command

The next step is weighted DeBERTa retraining:

```powershell
.\.venv\Scripts\python.exe src\models\train_weighted_transformer.py `
  --train data\processed\round4_residual_train.jsonl `
  --valid data\processed\lit_academic_poetry_valid.jsonl `
  --test data\processed\lit_academic_poetry_internal_test.jsonl `
  --guard_dev data\processed\round4_residual_dev_hardpos.jsonl `
  --output_dir outputs\models\round4_deberta_weighted_residual `
  --model_name microsoft/deberta-v3-base `
  --epochs 3 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --learning_rate 1e-5 `
  --gradient_accumulation_steps 2 `
  --sample_weight_field sample_weight `
  --class_weight none `
  --fp16 `
  --seed 20260522
```

Pre-teacher gate after training:

```text
internal-test F1 >= Step7 preferred
hard-positive dev recall improves over Step7
hard-human mirror dev FP does not increase
```

Do not evaluate a Round4 teacher-test candidate until these non-teacher gates
are checked.

## 5. Step 3 Prepared: Human-Style Guard

New script:

```text
src/models/train_round4_human_style_guard.py
```

Command used:

```powershell
.\.venv\Scripts\python.exe src\models\train_round4_human_style_guard.py --threshold 0.75
```

Outputs:

```text
outputs/models/round4_human_style_guard/human_style_guard.pkl
outputs/models/round4_human_style_guard/human_style_guard_report.json
outputs/evaluation/round4_human_style_guard_report.md
```

Guard inference script:

```text
src/evaluation/predict_round4_human_style_guard.py
```

Internal-test guard check:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_round4_human_style_guard.py `
  --input data\processed\lit_academic_poetry_internal_test.jsonl `
  --output outputs\predictions\round4_human_style_guard_internal_test_predictions.jsonl `
  --metrics outputs\predictions\round4_human_style_guard_internal_test_metrics.json
```

Internal-test result:

| Split | Veto Rate | Human-Style Recall | Confusion |
| --- | ---: | ---: | --- |
| internal_test | 0.0116 | 0.0158 | [[841, 6], [870, 14]] |

Interpretation:

```text
Positive guard label = high-style human.
Use this only as a veto signal for unsafe human -> LLM overrides.
Do not use it as a global detector.
```

Threshold sweep favored `0.75` as a conservative veto point:

| Split | Veto Rate | Meaning |
| --- | ---: | --- |
| hard-positive dev | 0.0540 | only 5.4% of LLM hard positives are wrongly vetoed |
| hard-human dev | 0.2140 | 21.4% of hard human mirrors are protected |

At threshold `0.75`, the combined-dev human-style precision is `0.7985`, and
the hard-human-only precision is `1.0000` because every veto on that split is a
true human-style row. The guard is useful but intentionally limited: it should
reduce risky overrides, not decide the whole task.

Verification:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\models\train_round4_human_style_guard.py
```

Result:

```text
passed
```
