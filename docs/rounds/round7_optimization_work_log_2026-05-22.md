# Round7 Optimization Work Log

Date: 2026-05-22

Round7 starts from:

```text
docs/rounds/round6_detailed_work_record_and_round7_plan_2026-05-22.md
```

The strict boundary is unchanged:

```text
data/raw/teacher_test.json is diagnostic-only.
Teacher-test was not used for Round7 audit, candidate mining, dataset splitting,
selector training, selector threshold selection, or override rule search.
```

Current decision after this Round7 block:

```text
PROMOTE_TO_TEACHER_TEST = completed
FINAL_MODEL_CANDIDATE = no
KEEP_FINAL_MODEL = Step7 ensemble
```

## 1. Phase 0 Exact-Candidate Audit

New script:

```text
src/evaluation/audit_round7_exact_candidates.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\audit_round7_exact_candidates.py
```

Outputs:

```text
outputs/evaluation/round7_exact_candidate_audit.json
outputs/evaluation/round7_exact_candidate_audit.md
```

The audit reproduced the Round6 failure mode on the exact disagreement surface:

| Split | Exact safe total | Exact unsafe total | Exact unsafe blocked by Round6 |
| --- | ---: | ---: | ---: |
| internal_test | 16 | 35 | 2 |

Interpretation:

```text
Round6 proxy dev success did not transfer to exact Step7-human -> Round4-LLM
unsafe cases. Round7 should train and gate on exact disagreement candidates.
```

## 2. Phase 1 Exact-Candidate Dataset

New script:

```text
src/data/build_round7_exact_candidate_dataset.py
```

The first dataset pass used only Round5 `hardpos` / `hardneg` exact candidates:

```text
pool = 120 = 88 safe + 32 unsafe
PROMOTE_TO_ROUND7_SELECTOR_TRAINING = no
```

To fill the exact-candidate pool without touching teacher-test, Round7 added:

```text
src/evaluation/mine_round7_train_exact_candidates.py
```

Prediction and mining sources:

| Source | Use |
| --- | --- |
| `data/processed/round4_residual_train.jsonl` | train-side Step7-vs-Round4 disagreement mine |
| `data/processed/round4_old_prose_human_mirror_candidates.jsonl` | unused old-prose human unsafe mine |

Main mining outputs:

```text
data/processed/round7_round4_train_all_exact_like_mined.jsonl
data/processed/round7_old_prose_human_exact_like_mined.jsonl
outputs/evaluation/round7_round4_train_all_exact_like_mine_report.md
outputs/evaluation/round7_old_prose_human_exact_like_mine_report.md
```

Final dataset command:

```powershell
.\.venv\Scripts\python.exe src\data\build_round7_exact_candidate_dataset.py `
  --extra_exact_pool data\processed\round7_round4_train_all_exact_like_mined.jsonl `
  --extra_exact_pool data\processed\round7_old_prose_human_exact_like_mined.jsonl
```

Final dataset outputs:

```text
data/processed/round7_exact_candidate_train.jsonl
data/processed/round7_exact_candidate_dev.jsonl
data/processed/round7_exact_candidate_dataset_report.json
data/processed/round7_exact_candidate_dataset_report.md
```

Final count and leakage gate:

| Split | Rows | Safe | Unsafe |
| --- | ---: | ---: | ---: |
| train | 769 | 359 | 410 |
| dev | 202 | 82 | 120 |

| Check | Count |
| --- | ---: |
| train/dev group overlap | 0 |
| train/dev text overlap | 0 |
| teacher-test exact text duplicates | 0 |
| held-out probe/train text overlap | 0 |
| held-out probe/dev text overlap | 0 |

Dataset decision:

```text
PROMOTE_TO_ROUND7_SELECTOR_TRAINING = yes
```

## 3. Phase 2 Exact Selector Baseline

New scripts:

```text
src/models/train_round7_exact_candidate_selector.py
src/evaluation/predict_round7_exact_candidate_selector.py
```

Training command:

```powershell
.\.venv\Scripts\python.exe src\models\train_round7_exact_candidate_selector.py
```

Model:

```text
LogisticRegression
word TF-IDF + char TF-IDF + probability / bucket / text-shape features
positive label = safe_override
threshold selected on exact dev only
```

Outputs:

```text
outputs/models/round7_exact_candidate_selector/selector.pkl
outputs/models/round7_exact_candidate_selector/selector_report.json
outputs/evaluation/round7_exact_candidate_selector_report.md
outputs/predictions/round7_exact_selector_train_predictions.jsonl
outputs/predictions/round7_exact_selector_dev_predictions.jsonl
outputs/predictions/round7_exact_selector_probe_mixed_predictions.jsonl
```

Selected threshold:

```text
p_round7_safe_override >= 0.6700
```

Selector metrics:

| Split | Safe pass | Unsafe block |
| --- | ---: | ---: |
| exact dev | 0.4634 | 0.9083 |
| held-out internal-style probe | 0.5625 | 0.8000 |

Held-out probe comparison:

| Selector | Unsafe blocked |
| --- | ---: |
| Round6 v1b | 2 / 35 |
| Round7 baseline | 28 / 35 |

Selector decision:

```text
PROMOTE_TO_ROUND7_RULE_SEARCH = yes
```

## 4. Phase 3 Two-Stage Override Rule Search

New script:

```text
src/evaluation/tune_round7_exact_safe_override.py
```

Prediction commands:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_round7_exact_candidate_selector.py `
  --input outputs\predictions\round6_safe_selector_internal_test_predictions.jsonl `
  --output outputs\predictions\round7_exact_selector_internal_test_predictions.jsonl

.\.venv\Scripts\python.exe src\evaluation\predict_round7_exact_candidate_selector.py `
  --input outputs\predictions\round6_safe_selector_hardpos_predictions.jsonl `
  --output outputs\predictions\round7_exact_selector_hardpos_predictions.jsonl

.\.venv\Scripts\python.exe src\evaluation\predict_round7_exact_candidate_selector.py `
  --input outputs\predictions\round6_safe_selector_hardneg_predictions.jsonl `
  --output outputs\predictions\round7_exact_selector_hardneg_predictions.jsonl
```

Rule-search command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\tune_round7_exact_safe_override.py `
  --tune_set internal_test outputs\predictions\round7_exact_selector_internal_test_predictions.jsonl `
  --tune_set hardpos outputs\predictions\round7_exact_selector_hardpos_predictions.jsonl `
  --tune_set hardneg outputs\predictions\round7_exact_selector_hardneg_predictions.jsonl
```

Outputs:

```text
outputs/models/round7_exact_safe_override/rules.json
outputs/models/round7_exact_safe_override/tuning_report.json
outputs/evaluation/round7_exact_safe_override_tuning_report.md
outputs/predictions/round7_exact_safe_override_internal_test_predictions.jsonl
outputs/predictions/round7_exact_safe_override_hardpos_predictions.jsonl
outputs/predictions/round7_exact_safe_override_hardneg_predictions.jsonl
```

Search result:

| Item | Result |
| --- | ---: |
| candidate rules | 9217 |
| feasible rules | 256 |
| selected overrides | 72 |

Selected non-teacher result:

| Split | Step7 FP | Round7 FP | Step7 FN | Round7 FN | Fixed Step7 FN | Induced FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 27 | 27 | 46 | 36 | 10 | 0 |
| hardpos | 0 | 0 | 292 | 230 | 62 | 0 |
| hardneg | 26 | 26 | 0 | 0 | 0 | 0 |

Rule-search decision:

```text
PROMOTE_TO_ROUND7_GATE_REPORT = yes
```

## 5. Gate Report And Next Step

Gate report:

```text
outputs/evaluation/round7_gate_report.md
```

Round7 has now cleared the planned non-teacher gates:

| Gate | Result |
| --- | --- |
| hardneg induced FP | 0 |
| internal induced FP | 0 |
| internal F1 | 0.9626 |
| hardpos fixed Step7 FN | 62 |
| held-out exact probe unsafe block | 28 / 35, improved from 2 / 35 |

Next step:

```text
Run one frozen Round7 teacher-test diagnostic.
If Round7 does not beat Step7 on teacher-test, keep Step7 as the final model.
```

## 6. Frozen Teacher-Test Diagnostic

New script:

```text
src/evaluation/apply_round7_exact_safe_override.py
```

Teacher-test scoring command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_round7_exact_candidate_selector.py `
  --input outputs\predictions\round5_flip_guard_teacher_test_predictions.jsonl `
  --output outputs\predictions\round7_exact_selector_teacher_test_predictions.jsonl
```

Frozen rule application:

```powershell
.\.venv\Scripts\python.exe src\evaluation\apply_round7_exact_safe_override.py `
  --input outputs\predictions\round7_exact_selector_teacher_test_predictions.jsonl
```

Outputs:

```text
outputs/predictions/round7_exact_selector_teacher_test_predictions.jsonl
outputs/predictions/round7_teacher_test_predictions.jsonl
outputs/evaluation/round7_teacher_test_comparison.json
outputs/evaluation/round7_teacher_test_comparison.md
outputs/evaluation/round7_teacher_test_ledger_summary.json
docs/rounds/round7_final_decision_2026-05-22.md
```

Teacher-test result:

| Run | Correct / 300 | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Step7 baseline | 274 | 0.9133 | 0.9133 | 13 | 13 |
| Round7 exact override | 274 | 0.9133 | 0.9133 | 13 | 13 |

Teacher-test override surface:

| Candidate type | Count |
| --- | ---: |
| safe fixed-FN candidate | 3 |
| unsafe induced-FP candidate | 14 |
| selected Round7 overrides | 0 |

Final decision:

```text
PROMOTE_AS_FINAL = no
KEEP_FINAL_MODEL = Step7 ensemble
```

The frozen Round7 rule remains conservative on teacher-test. The one safe
`general_prose` candidate fails the strict general unsafe guard, and the two
safe `literary_short_fragment` candidates fail the base unsafe guard. That
keeps Round7 FP-safe but makes the final diagnostic a no-op.
