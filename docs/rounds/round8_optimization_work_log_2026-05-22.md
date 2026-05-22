# Round8-OneShot Optimization Work Log

Date: 2026-05-22

This log records the first execution pass of the final Round8-OneShot route.
It follows `docs/rounds/round8_one_shot_95_optimization_plan_2026-05-22.md`.

Strict boundary:

```text
Teacher-test was not used for training, threshold tuning, selector calibration,
rule search, router tuning, model selection, or promotion.
Teacher-test text was used only for exact duplicate exclusion in data reports.
```

## 1. Documents Reviewed

Read:

```text
PROJECT_REPORT.md
README.md
docs/rounds/round8_one_shot_95_optimization_plan_2026-05-22.md
docs/rounds/round2_results_summary_2026-05-22.md
docs/rounds/round2_postmortem_and_round3_plan_2026-05-22.md
docs/rounds/round3_results_summary_2026-05-22.md
docs/rounds/round4_v1_summary_and_round5_plan_2026-05-22.md
docs/rounds/round5_final_decision_2026-05-22.md
docs/rounds/round5_supplement_and_round6_plan_2026-05-22.md
docs/rounds/round6_detailed_work_record_and_round7_plan_2026-05-22.md
docs/rounds/round7_final_decision_2026-05-22.md
docs/rounds/round7_detailed_work_record_and_round8_plan_2026-05-22.md
```

Cross-round conclusion:

```text
Step7 remains the final baseline.
Round2-Round7 showed that router/rule/guard tweaks either regress teacher-test
or become safe no-ops.
Round8 must be data-first: residual distribution rebuild, residual-aware model,
stylometry signal, and local ambiguous-zone selection.
```

## 2. Phase 1: Residual Taxonomy

Created:

```text
docs/rounds/round8_residual_error_taxonomy_2026-05-22.md
```

It defines 16 residual buckets:

```text
8 human hard-negative buckets
8 LLM hard-positive buckets
```

It also records metadata requirements, split policy, leakage checks, and
promotion gates.

## 3. Phase 2: Residual Candidate Pool

Created script:

```text
src/data/build_residual_candidate_pool.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\data\build_residual_candidate_pool.py
```

Outputs:

```text
data/processed/residual_candidate_pool_v1.jsonl
data/processed/residual_candidate_pool_v1_report.json
```

Result:

| Item | Value |
| --- | ---: |
| total candidates | 7,651 |
| human hard negatives | 4,040 |
| LLM hard positives | 3,611 |
| split groups | 5,224 |
| taxonomy buckets present | 16 / 16 |
| teacher-test exact duplicates | 0 |
| ready for Step7 scoring | yes |

## 4. Phase 3: Step7 Scoring

Created script:

```text
src/evaluation/predict_step7_on_residual_candidates.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_step7_on_residual_candidates.py --batch_size 32
```

Outputs:

```text
outputs/predictions/residual_candidate_pool_v1_step7_predictions.jsonl
outputs/predictions/residual_candidate_pool_v1_step7_metrics.json
```

Frozen Step7:

```text
TF-IDF:  outputs/models/tfidf_lit_academic_poetry
DeBERTa: outputs/models/deberta_lit_academic_poetry_step7_combined
alpha:   0.5
threshold: 0.55
```

Residual-pool Step7 result:

| Accuracy | Precision | Recall | F1 | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8544 | 0.9664 | 0.7164 | 0.8228 | 90 | 1,024 |

This confirms the pool exposes hard LLM false negatives while preserving a
useful human hard-negative safety surface.

## 5. Phase 4: Hard Residual Split

Created script:

```text
src/data/select_step7_hard_residuals.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\data\select_step7_hard_residuals.py
```

Outputs:

```text
data/processed/residual_selected_v1.jsonl
data/processed/residual_train_v1.jsonl
data/processed/residual_dev_v1.jsonl
data/processed/residual_probe_v1.jsonl
data/processed/residual_split_v1_report.json
```

Split summary:

| Split | Rows | Human | LLM | Step7 Error Rows | Core Hard Rows | Ambiguous Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_train_v1 | 3,064 | 1,264 | 1,800 | 523 | 463 | 320 |
| residual_dev_v1 | 800 | 400 | 400 | 370 | 367 | 28 |
| residual_probe_v1 | 426 | 226 | 200 | 221 | 217 | 59 |

Leakage:

```text
train/dev/probe group overlap = 0
train/dev/probe text overlap = 0
teacher-test exact duplicate = 0
```

Decision:

```text
ready_for_residual_mix_train_build = yes
```

Important caveat:

```text
The split is good for residual DeBERTa training, but residual_dev has only 28
ambiguous-zone rows. A dedicated ambiguous selector dataset or re-split is
needed before thresholding an ambiguous-zone selector.
```

## 6. Phase 5: Residual Mix Train

Created script:

```text
src/data/build_round8_residual_mix_train.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\data\build_round8_residual_mix_train.py
```

Outputs:

```text
data/processed/lit_academic_poetry_train_round8_residual_mix.jsonl
data/processed/lit_academic_poetry_train_round8_residual_mix_report.json
```

Result:

| Item | Value |
| --- | ---: |
| total rows | 10,213 |
| base rows | 7,149 |
| residual rows | 3,064 |
| residual ratio | 0.3000 |
| ready for DeBERTa training | yes |

## 7. Phase 6: Residual-Aware DeBERTa V1

Training command:

```powershell
.\.venv\Scripts\python.exe src\models\train_deberta.py `
  --train data\processed\lit_academic_poetry_train_round8_residual_mix.jsonl `
  --valid data\processed\lit_academic_poetry_valid.jsonl `
  --test data\processed\lit_academic_poetry_internal_test.jsonl `
  --output_dir outputs\models\deberta_round8_residual_mix `
  --model_name outputs\models\deberta_lit_academic_poetry_step7_combined\best_model `
  --max_length 512 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --learning_rate 5e-6 `
  --epochs 2 `
  --weight_decay 0.01 `
  --warmup_ratio 0.1 `
  --fp16 `
  --seed 20260522
```

Outputs:

```text
outputs/models/deberta_round8_residual_mix/
```

Original-split metrics at threshold 0.5:

| Split | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| valid | 0.9398 | 0.9015 | 0.9846 | 0.9412 |
| internal_test | 0.9417 | 0.9063 | 0.9823 | 0.9428 |

Residual-split metrics at threshold 0.5:

| Split | Accuracy | Precision | Recall | F1 | Confusion |
| --- | ---: | ---: | ---: | ---: | --- |
| residual_dev_v1 | 0.7325 | 0.7853 | 0.6400 | 0.7052 | `[[330, 70], [144, 256]]` |
| residual_probe_v1 | 0.6033 | 0.5742 | 0.6000 | 0.5868 | `[[137, 89], [80, 120]]` |

Threshold sweep diagnostics:

| Split | Best F1 Threshold | Best F1 |
| --- | ---: | ---: |
| valid | 0.94 | 0.9472 |
| internal_test | 0.84 | 0.9501 |
| residual_dev_v1 | 0.05 | 0.7627 |
| residual_probe_v1 | 0.05 | 0.6416 |

Comparison against Step7 on residual surfaces:

| Split | Step7 F1 | Round8 Residual DeBERTa Best Diagnostic F1 |
| --- | ---: | ---: |
| residual_dev_v1 | 0.2060 | 0.7627 |
| residual_probe_v1 | 0.1723 | 0.6416 |

Decision:

```text
GLOBAL_PROMOTION = no
USE_AS_SIGNAL_BRANCH = yes
```

Reason:

```text
The residual-aware DeBERTa learned useful hard-positive signal, but it hurts
original internal-test F1 relative to Step7. It cannot replace Step7 globally.
It should be used only inside a local selector/fusion system.
```

## 8. Phase 7: Stylometry Branch

Created scripts:

```text
src/models/train_stylometry_branch.py
src/evaluation/predict_stylometry_branch.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\models\train_stylometry_branch.py
```

Outputs:

```text
outputs/models/stylometry_round8/stylometry_branch.pkl
outputs/models/stylometry_round8/stylometry_branch_report.json
outputs/models/stylometry_round8/predictions/
```

Metrics at threshold 0.5:

| Split | Precision | Recall | F1 | Confusion |
| --- | ---: | ---: | ---: | --- |
| train | 0.9641 | 0.9304 | 0.9470 | `[[4862, 179], [360, 4812]]` |
| valid | 0.8995 | 0.8783 | 0.8888 | `[[799, 83], [103, 743]]` |
| internal_test | 0.9147 | 0.8607 | 0.8869 | `[[816, 68], [118, 729]]` |
| residual_dev_v1 | 0.6300 | 0.4725 | 0.5400 | `[[289, 111], [211, 189]]` |
| residual_probe_v1 | 0.5424 | 0.4800 | 0.5093 | `[[145, 81], [104, 96]]` |

Decision:

```text
GLOBAL_PROMOTION = no
USE_AS_SELECTOR_FEATURE = yes
```

Reason:

```text
Stylometry is not strong enough as a standalone detector. Its useful role is as
an independent surface-style feature for the ambiguous-zone selector.
```

## 9. Current State

Completed:

```text
Phase 1 taxonomy
Phase 2 candidate pool
Phase 3 Step7 scoring
Phase 4 hard residual split
Phase 5 70/30 train mix
Phase 6 residual-aware DeBERTa V1
Phase 7 stylometry branch V1
```

Current candidate status:

```text
Step7 remains final baseline.
Round8 residual DeBERTa V1 is a useful signal branch, not a global model.
Stylometry V1 is a selector feature, not a global model.
No teacher-test diagnostic is allowed at this stage.
```

## 10. Next Step

Do not train the ambiguous selector directly on the current residual split
without addressing the ambiguous-dev sample count.

Recommended next implementation:

```text
1. Build data/processed/round8_ambiguous_selector_{train,dev,probe}.jsonl
   from all non-teacher rows where 0.35<=p_step7<=0.65.
2. Keep group leakage checks and teacher exact duplicate = 0.
3. Score residual DeBERTa and stylometry on the ambiguous selector splits.
4. Train src/models/train_round8_ambiguous_selector.py.
5. Gate the selector on original internal_test + residual_dev/probe before any
   teacher-test diagnostic.
```

## 11. Phase 8: Ambiguous Selector Follow-up

Implemented after the Phase 7 caveat that `residual_dev_v1` had only 28
ambiguous-zone rows.

Created scripts:

```text
src/data/build_round8_ambiguous_selector_splits.py
src/evaluation/score_round8_ambiguous_selector_splits.py
src/models/train_round8_ambiguous_selector.py
src/evaluation/predict_round8_oneshot_system.py
src/evaluation/evaluate_round8_gate.py
```

Dedicated ambiguous selector split:

```text
data/processed/round8_ambiguous_selector_train.jsonl
data/processed/round8_ambiguous_selector_dev.jsonl
data/processed/round8_ambiguous_selector_probe.jsonl
data/processed/round8_ambiguous_selector_split_report.json
```

Split summary:

| Split | Rows | Human | LLM | Step7 Errors | Teacher Exact Duplicates |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 244 | 63 | 181 | 118 | 0 |
| dev | 81 | 21 | 60 | 47 | 0 |
| probe | 82 | 21 | 61 | 38 | 0 |

Leakage:

```text
train/dev/probe group overlap = 0
train/dev/probe text overlap = 0
teacher-test exact duplicate = 0
```

Scored outputs:

```text
outputs/predictions/round8_ambiguous_selector_train_scored.jsonl
outputs/predictions/round8_ambiguous_selector_dev_scored.jsonl
outputs/predictions/round8_ambiguous_selector_probe_scored.jsonl
outputs/predictions/round8_ambiguous_selector_score_report.json
```

Branch diagnostics on dedicated ambiguous splits:

| Split | Step7 F1 | Residual DeBERTa F1 | Stylometry F1 |
| --- | ---: | ---: | ---: |
| dev | 0.5053 | 0.8872 | 0.8596 |
| probe | 0.6122 | 0.8722 | 0.8908 |

Selector audit note:

```text
An initial selector run produced impossible 1.0000 dev/probe metrics because
label-derived metadata fields leaked into features. The final selector removes
round8_bucket, round8_bucket_family, generator, and selection_tier from the
feature dict before training.
```

Final ambiguous selector:

```text
outputs/models/round8_ambiguous_selector/selector.pkl
outputs/models/round8_ambiguous_selector/selector_report.json
outputs/evaluation/round8_ambiguous_selector_report.md
```

Local ambiguous split result at selected confidence threshold `0.63`:

| Split | Step7 F1 | Local Round8 F1 | Net Correct Gain | New FP |
| --- | ---: | ---: | ---: | ---: |
| dev | 0.5053 | 0.9298 | +39 | 0 |
| probe | 0.6122 | 0.9231 | +29 | 0 |

Wider non-teacher gate result:

```text
docs/rounds/round8_one_shot_gate_report_2026-05-22.md
```

| Split | Step7 F1 | Round8 F1 | F1 Delta | Net Correct | New FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 0.9570 | +0.0006 | +1 | 4 |
| residual_dev | 0.2060 | 0.2363 | +0.0303 | +8 | 0 |
| residual_probe | 0.1723 | 0.2406 | +0.0683 | +19 | 3 |

Decision:

```text
ROUND8_TEACHER_TEST_DIAGNOSTIC_ALLOWED = no.
Step7 remains the final baseline. The ambiguous selector is a reusable
diagnostic/signal artifact, but it is not promoted for teacher-test submission.
```
