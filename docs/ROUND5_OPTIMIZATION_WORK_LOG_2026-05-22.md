# Round5 Optimization Work Log

Date: 2026-05-22

Round5 follows `docs/ROUND4_V1_SUMMARY_AND_ROUND5_PLAN_2026-05-22.md`.
The route remains strict: `data/raw/teacher_test.json` is not used for
training, threshold selection, guard tuning, rule tuning, or model selection.

## 0. Repository Orientation

Initial file traversal confirmed that the repository now contains Round2,
Round3, Round4, and Round5-ready artifacts under:

```text
src/data/
src/models/
src/evaluation/
data/processed/
docs/
outputs/
```

Important current state:

- `src/evaluation/predict_ensemble.py` and `src/evaluation/predict_neural_model.py` already had local Round4 metadata-preservation edits before this Round5 block.
- Round4 generated data and scripts are present but still untracked in git.
- The Round4 DeBERTa prediction files exist, but the model artifact
  `outputs/models/round4_deberta_weighted_residual/best_model` is not present locally.

## 1. Phase 0 Completed: Baseline Freeze

New script:

```text
src/evaluation/build_round5_flip_ledger.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\build_round5_flip_ledger.py
```

Outputs:

```text
outputs/evaluation/round5_baseline_frozen_report.md
outputs/evaluation/round5_baseline_frozen_report.json
```

Frozen metrics:

| Split | Run | F1 | FP | FN |
| --- | --- | ---: | ---: | ---: |
| internal_test | Step7 | 0.9564 | 27 | 46 |
| internal_test | Round4 DeBERTa | 0.9435 | 60 | 37 |
| hardpos | Step7 | 0.5876 | 0 | 292 |
| hardpos | Round4 DeBERTa | 0.7229 | 0 | 217 |
| hardneg | Step7 | 0.0000 | 26 | 0 |
| hardneg | Round4 DeBERTa | 0.0000 | 53 | 0 |

Note: the Round5 plan text expected Round4 internal FP 59 / F1 0.9441, but the
current frozen prediction file reports FP 60 / F1 0.9435. Round5 uses the
actual prediction file as the source of truth.

Alignment:

| Split | Aligned rows | Missing Round4 | Missing Guard | Pair mismatch |
| --- | ---: | ---: | ---: | ---: |
| internal_test | 1731 | 0 | 0 | 0 |
| hardpos | 500 | 0 | 0 | 0 |
| hardneg | 500 | 0 | 0 | 0 |

## 2. Phase 1 Completed: Flip Ledger

Outputs:

```text
outputs/evaluation/round5_flip_ledger.jsonl
outputs/evaluation/round5_flip_ledger_summary.md
outputs/evaluation/round5_flip_ledger_summary.json
data/processed/round5_flip_guard_train.jsonl
data/processed/round5_flip_guard_dev_hardpos.jsonl
data/processed/round5_flip_guard_dev_hardneg.jsonl
```

Core counts:

| Split | Safe fixed-FN candidates | Unsafe induced-FP candidates | Total override candidates |
| --- | ---: | ---: | ---: |
| internal_test | 16 | 35 | 51 |
| hardpos | 88 | 0 | 88 |
| hardneg | 0 | 32 | 32 |

Interpretation:

- There is enough signal to attempt a lightweight flip guard, but the initial
  train set is small and imbalanced.
- Hardpos repair candidates are concentrated in `literary_short_fragment`,
  `literary_old_prose`, and `poetry_freeverse`.
- Induced FP risk is broad, with many `general_prose` and `literary_old_prose`
  cases, so the override rule should not be global.

## 3. Phase 3 Completed: Flip Guard

New scripts:

```text
src/models/train_round5_flip_guard.py
src/evaluation/predict_round5_flip_guard.py
```

Training command:

```powershell
.\.venv\Scripts\python.exe src\models\train_round5_flip_guard.py
```

Outputs:

```text
outputs/models/round5_flip_guard/flip_guard.pkl
outputs/models/round5_flip_guard/flip_guard_report.json
outputs/evaluation/round5_flip_guard_report.md
outputs/predictions/round5_flip_guard_internal_test_predictions.jsonl
outputs/predictions/round5_flip_guard_hardpos_predictions.jsonl
outputs/predictions/round5_flip_guard_hardneg_predictions.jsonl
```

Selected threshold:

```text
p_unsafe_override >= 0.29 means veto unsafe override
```

Gate result:

| Gate | Required | Observed | Pass |
| --- | --- | ---: | --- |
| hardneg induced-FP protection | >= 0.70 | 0.8750 | yes |
| hardpos safe-override veto | <= 0.10 | 0.0795 | yes |
| internal candidate-veto rate | <= 0.03 | 0.0202 | yes |

Important caveat: the flip guard is intended only for Step7-human -> Round4-LLM
override candidates. It should not be interpreted as a global detector.

## 4. Phase 5 Completed: Residual Override Search

New scripts:

```text
src/evaluation/tune_round5_residual_override.py
src/evaluation/apply_round5_residual_override.py
```

The first full-grid shell call exceeded the foreground timeout, but the process
completed and wrote the current frozen rule/report. A reduced-grid rerun also
found a safe rule; the full-grid output below is the stronger current freeze:

```powershell
.\.venv\Scripts\python.exe src\evaluation\tune_round5_residual_override.py `
  --round5_thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 `
  --min_deltas 0.00,0.05,0.10,0.15,0.20,0.25 `
  --flip_guard_unsafe_max 0.20,0.25,0.29,0.30,0.35,0.40 `
  --human_style_veto_max 0.65,0.70,0.75,0.80 `
  --min_words 0,16,32,48 `
  --bucket_groups all,planned_high_risk,literary_poetry,old_short,poetry,academic `
  --tune_set internal_test step7=outputs\predictions\round4_step7_internal_test_predictions.jsonl round5=outputs\predictions\round4_deberta_internal_test_predictions.jsonl human_guard=outputs\predictions\round4_human_style_guard_internal_test_predictions.jsonl flip_guard=outputs\predictions\round5_flip_guard_internal_test_predictions.jsonl `
  --tune_set hardpos step7=outputs\predictions\round4_step7_hardpos_predictions.jsonl round5=outputs\predictions\round4_deberta_hardpos_predictions.jsonl human_guard=outputs\predictions\round4_human_style_guard_hardpos_predictions.jsonl flip_guard=outputs\predictions\round5_flip_guard_hardpos_predictions.jsonl `
  --tune_set hardneg step7=outputs\predictions\round4_step7_hardneg_predictions.jsonl round5=outputs\predictions\round4_deberta_hardneg_predictions.jsonl human_guard=outputs\predictions\round4_human_style_guard_hardneg_predictions.jsonl flip_guard=outputs\predictions\round5_flip_guard_hardneg_predictions.jsonl
```

Selected rule:

```json
{
  "round5_threshold": 0.55,
  "min_delta": 0.0,
  "flip_guard_unsafe_max": 0.35,
  "human_style_veto_max": 0.8,
  "min_words": 0,
  "bucket_group": "old_short",
  "allowed_buckets": [
    "literary_old_prose",
    "literary_short_fragment"
  ],
  "disabled_baseline": false
}
```

Gate metrics:

| Split | Step7 F1 | Round5 F1 | Step7 FP | Round5 FP | Step7 FN | Round5 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 0.9570 | 27 | 27 | 46 | 45 |
| hardpos | 0.5876 | 0.6928 | 0 | 0 | 292 | 235 |
| hardneg | 0.0000 | 0.0000 | 26 | 26 | 0 | 0 |

Override delta:

| Split | Overrides | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: |
| internal_test | 1 | 1 | 0 |
| hardpos | 57 | 57 | 0 |
| hardneg | 0 | 0 | 0 |

Decision:

```text
PROMOTE_TO_PHASE6_GATE_REPORT = yes
```

## 5. Phase 6 Completed: Gate Report

Output:

```text
outputs/evaluation/round5_gate_report.md
```

Promotion decision:

```text
PROMOTE_TO_TEACHER_TEST = yes
FINAL_MODEL_CANDIDATE = no
REASON = Non-teacher gates pass, but final promotion still requires Phase 7 teacher-test diagnostic.
```

## 6. Phase 7 Completed: Teacher-Test Diagnostic

The missing Round4 DeBERTa branch was retrained:

```text
outputs/models/round4_deberta_weighted_residual/best_model
```

Retraining command:

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

Retrained branch metrics:

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.9531 | 0.9443 | 0.9610 | 0.9525 | 48 | 33 |
| internal_test | 0.9434 | 0.9241 | 0.9634 | 0.9434 | 67 | 31 |
| hardpos dev | 0.5920 | 1.0000 | 0.5920 | 0.7437 | 0 | 204 |

Teacher-test branch result:

```text
Round4 DeBERTa branch: 263 / 300, accuracy 0.8767, FP 23, FN 14
```

Final teacher-test application:

```powershell
.\.venv\Scripts\python.exe src\evaluation\apply_round5_residual_override.py `
  --split_name teacher_test `
  --step7 outputs\predictions\round5_step7_teacher_test_predictions.jsonl `
  --round5 outputs\predictions\round5_round4_deberta_teacher_test_predictions.jsonl `
  --human_guard outputs\predictions\round5_human_style_guard_teacher_test_predictions.jsonl `
  --flip_guard outputs\predictions\round5_flip_guard_teacher_test_predictions.jsonl `
  --rules outputs\models\round5_residual_override\rules.json `
  --output outputs\predictions\round5_teacher_test_predictions.jsonl `
  --metrics outputs\evaluation\round5_teacher_test_comparison.json `
  --report_md outputs\evaluation\round5_teacher_test_comparison.md
```

Final result:

| Run | Correct / 300 | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step7 baseline | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |
| Round5 override | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |

Override delta:

| Overrides | Fixed Step7 FN | Induced FP | Broke Step7 correct |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |

Teacher-test flip ledger:

| Type | Count |
| --- | ---: |
| safe fixed-FN candidate | 3 |
| unsafe induced-FP candidate | 14 |
| total Step7-human -> Round4-LLM candidates | 17 |

Why no override fired:

- one safe candidate was `general_prose`, outside the frozen `old_short` bucket group;
- two safe `literary_short_fragment` candidates had `p_unsafe_override > 0.35`;
- the same constraints correctly blocked all induced-FP candidates.

Final decision:

```text
PROMOTE_AS_FINAL = no
KEEP_FINAL_MODEL = Step7 ensemble
```

Saved:

```text
outputs/predictions/round5_teacher_test_predictions.jsonl
outputs/evaluation/round5_teacher_test_comparison.md
docs/ROUND5_FINAL_DECISION_2026-05-22.md
```

## 7. Supplement and Next-Round Handoff

Additional summary and the detailed Round6 execution plan were written to:

```text
docs/ROUND5_SUPPLEMENT_AND_ROUND6_PLAN_2026-05-22.md
```

The recommended next route is `Round6: Safe Override Distillation`, focused on
building non-teacher candidate-level safe/unsafe override data and training a
selector that can release safe short/general-prose overrides without increasing
hard-human false positives.
