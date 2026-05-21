# Round3 Startup Readiness

Updated: 2026-05-21

This file records the working state before starting the third optimization
round. It is intentionally short and operational. The source of truth for the
strategy remains `docs/ROUND2_POSTMORTEM_AND_ROUND3_PLAN.md`.

## 1. Current Baseline

The strict-route final model is still the Step7 DeBERTa + TF-IDF ensemble:

```text
DeBERTa: outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:  outputs/models/tfidf_lit_academic_poetry
alpha:   0.5
threshold: 0.55
```

Teacher-test diagnostic result:

```text
accuracy = 0.9133
correct = 274 / 300
FP = 13
FN = 13
confusion = [[137, 13], [13, 137]]
```

The 95% target needs at least `285 / 300`, so Round3 needs a net gain of
`+11` correct examples while preventing new human false positives.

## 2. Round3 Direction

Round3 should be a precision-guarded repair round, not a more aggressive
version of Round2.

Default policy:

```text
final_pred = step7_pred
```

New branches or stackers should only override Step7 when there is strong
evidence that the override fixes false negatives without creating uncontrolled
human false positives.

## 3. Ready Inputs

The files needed for Phase A are present locally:

```text
outputs/predictions/round2_step7_teacher_test_predictions.jsonl
outputs/predictions/round2_bucket_routed_teacher_test_predictions.jsonl
outputs/predictions/round2_stacker_teacher_test_predictions.jsonl
outputs/predictions/round2_roberta_teacher_test_predictions.jsonl
outputs/predictions/round2_stacker_with_roberta_teacher_test_predictions.jsonl
outputs/round2/error_ledger_teacher_step7.csv
outputs/round2/error_ledger_teacher_step7.jsonl
```

Round2 teacher-like data is also available:

```text
data/processed/round2_teacher_like_train.jsonl
data/processed/round2_teacher_like_dev.jsonl
data/processed/round2_teacher_like_report.json
```

## 4. Immediate First Task

Start with Phase A: Round2 error-delta audit.

Expected new script:

```text
src/evaluation/round3_error_delta_audit.py
```

Expected outputs:

```text
outputs/round3/error_delta_audit.csv
outputs/round3/error_delta_audit.md
outputs/round3/error_delta_by_bucket.json
```

The audit should answer:

1. Which Step7 false negatives were fixed by each Round2 candidate.
2. Which Step7-correct human samples were turned into new false positives.
3. Whether new false positives cluster in poetry, old prose, academic, or short
   fragment buckets.
4. Whether RoBERTa or stackers offer any low-risk override patterns.

## 5. Guardrails

Do not use teacher-test labels for training, threshold selection, router tuning,
stacker training, or model selection. Teacher-test labels are diagnostic only.

Do not promote a candidate just because it improves `round2_teacher_like_dev`.
Round2 showed that hard-dev gains can come with teacher-test human false
positive regressions.

Do not clean or revert the current working tree before checking with the user.
Round2 docs and scripts are currently present as local changes and are part of
the handoff context.

## 6. Verification Already Done

Environment check:

```text
.\.venv\Scripts\python.exe --version
Python 3.10.11
```

Lightweight code checks:

```text
.\.venv\Scripts\python.exe -m compileall src
git diff --check
```

Both checks completed without code errors. `git diff --check` only reported
line-ending warnings for existing modified files.

## 7. Suggested Next Command

After implementing the Phase A script, run it against the existing Round2
teacher-test candidate predictions:

```powershell
.\.venv\Scripts\python.exe src\evaluation\round3_error_delta_audit.py `
  --step7 outputs\predictions\round2_step7_teacher_test_predictions.jsonl `
  --candidates `
    bucket_routed=outputs\predictions\round2_bucket_routed_teacher_test_predictions.jsonl `
    stacker_step7=outputs\predictions\round2_stacker_teacher_test_predictions.jsonl `
    roberta_single=outputs\predictions\round2_roberta_teacher_test_predictions.jsonl `
    stacker_with_roberta=outputs\predictions\round2_stacker_with_roberta_teacher_test_predictions.jsonl `
  --output_csv outputs\round3\error_delta_audit.csv `
  --output_md outputs\round3\error_delta_audit.md `
  --output_json outputs\round3\error_delta_by_bucket.json
```
