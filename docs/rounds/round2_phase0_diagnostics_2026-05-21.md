# Round2 Phase 0 Diagnostics

Updated: 2026-05-21

This document records the Phase 0 workflow and results for the second-round
optimization pass. Phase 0 does not train a new model. Its purpose is to lock
the current Step7 baseline, identify the residual teacher-test errors, and
measure the diagnostic ceiling of the existing prediction family before adding
new data, a domain router, a stacker, or a third model branch.

## 1. Objective

The current final candidate is:

```text
DeBERTa:  outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:   outputs/models/tfidf_lit_academic_poetry
Fusion:   alpha = 0.5
Decision: threshold = 0.55
```

Teacher-test baseline:

| System | Accuracy | Correct | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | --- |
| Step7 ensemble | 0.9133 | 274 / 300 | 13 | 13 | `[[137, 13], [13, 137]]` |

The 95% target requires at least `285 / 300` correct predictions, so the
system must net-fix at least `11` of the current `26` teacher-test errors.

## 2. Scripts Added

Phase 0 added two reusable evaluation scripts:

```text
src/evaluation/export_error_ledger.py
src/evaluation/round2_threshold_family_diagnostics.py
```

### 2.1 Error Ledger Export

`export_error_ledger.py` builds a row-level ledger for the teacher-test
predictions. It joins prediction records with the raw teacher-test text and
extracts lightweight diagnostic features.

Important fields:

```text
id
label
prediction
error_type
probability
p_tfidf
p_deberta
text
length_chars
length_words
num_lines
linebreak_ratio
avg_line_length
punctuation_ratio
quote_count
dash_count
semicolon_count
archaic_word_count
academic_marker_count
rough_domain
confidence_bucket
notes
```

Command used:

```powershell
.\.venv\Scripts\python.exe src\evaluation\export_error_ledger.py `
  --predictions outputs\predictions\teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl `
  --input data\raw\teacher_test.json `
  --output_csv outputs\round2\error_ledger_teacher_step7.csv `
  --output_jsonl outputs\round2\error_ledger_teacher_step7.jsonl `
  --threshold 0.55
```

Generated outputs:

```text
outputs/round2/error_ledger_teacher_step7.csv
outputs/round2/error_ledger_teacher_step7.jsonl
```

### 2.2 Existing-Family Threshold Diagnostics

`round2_threshold_family_diagnostics.py` reads all existing teacher-test
prediction files and computes diagnostic-only oracle thresholds, error overlap,
and simple probability-average ensembles.

Important note:

```text
Oracle thresholds use teacher-test labels. They are diagnostic only and must
not be treated as valid model-selection or final-report tuning results.
```

Command used:

```powershell
.\.venv\Scripts\python.exe src\evaluation\round2_threshold_family_diagnostics.py `
  --predictions "outputs/predictions/teacher_test_*predictions.jsonl" `
  --output_md outputs\round2\existing_family_threshold_report.md `
  --output_json outputs\round2\existing_family_threshold_report.json `
  --max_combo_size 4 `
  --top_k 12
```

Generated outputs:

```text
outputs/round2/existing_family_threshold_report.md
outputs/round2/existing_family_threshold_report.json
```

## 3. Error Ledger Results

The ledger contains all `300` teacher-test rows.

| Error type | Count |
| --- | ---: |
| true_negative | 137 |
| true_positive | 137 |
| false_positive | 13 |
| false_negative | 13 |

The `26` residual errors have the following rough-domain distribution:

| Rough domain | Error count |
| --- | ---: |
| general_prose | 10 |
| poetry_freeverse | 10 |
| poetry_classical | 4 |
| literary_short_fragment | 1 |
| literary_old_prose | 1 |

Confidence-bucket distribution for the `26` errors:

| Confidence bucket | Error count | Meaning |
| --- | ---: | --- |
| near_boundary | 9 | These may be sensitive to thresholds, routing, or calibration |
| confident_human | 9 | These are mostly LLM false negatives with low LLM probability |
| confident_llm | 8 | These are mostly human false positives with high LLM probability |

This confirms that the remaining errors are not only boundary cases. A
meaningful subset is confidently wrong, so Phase 1 needs new teacher-like data
and later phases need routing or nonlinear fusion.

## 4. Threshold-Family Results

### 4.1 Single-Run Oracle Threshold Ceiling

Best single-run diagnostic ceiling:

| Run | Oracle threshold | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| step7_ensemble_combined_tfidf | 0.389250 | 0.9267 | 0.9286 | 9 | 13 |
| step7_ensemble_hardneg_poetry_tfidf | 0.391248 | 0.9267 | 0.9286 | 9 | 13 |
| step7_ensemble_hardneg_tfidf | 0.392845 | 0.9267 | 0.9286 | 9 | 13 |
| step7_ensemble_raw_tfidf | 0.668331 | 0.9267 | 0.9256 | 7 | 15 |

Interpretation: a single existing prediction file cannot reach 95%, even with
an oracle threshold chosen from teacher-test labels.

### 4.2 Simple Average Ensemble Ceiling

Best simple probability-average diagnostic result:

| Members | Oracle threshold | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| final_ensemble + step7_ensemble_hardneg_poetry_tfidf | 0.612190 | 0.9333 | 0.9324 | 8 | 12 |

Several closely related averages reach `0.9300`, but the best observed simple
average is still only `0.9333`. This is useful evidence that the current model
family does not contain enough independent signal to reach 95% through simple
probability averaging.

### 4.3 Error Overlap

The Step7 ensemble variants have identical current-decision errors. Their
pairwise overlap is:

```text
26 overlapping errors, Jaccard = 1.0000
```

They also overlap heavily with Step7 DeBERTa decisions:

```text
step7_deberta_calibrated vs Step7 ensemble variants:
26 overlapping errors, Jaccard = 0.9286
```

The older final ensemble has a different operating point and overlaps less
with the Step7 family:

```text
old_final_ensemble vs Step7 ensemble variants:
20 overlapping errors, Jaccard = 0.5714
```

This explains why averaging the old and new families helps somewhat, but the
ceiling is still below the 95% target.

## 5. Hard Cases

Wrong in all current decisions:

```text
5, 32, 38, 57, 73, 101, 112, 141, 171, 181, 189, 197,
209, 249, 264, 266, 285, 287, 292
```

Wrong in all best-accuracy oracle-threshold decisions:

```text
5, 32, 38, 57, 73, 171, 197, 209, 246, 249, 264, 287, 292
```

These IDs should drive the first teacher-like data buckets. They represent
errors that threshold tuning alone cannot consistently repair.

## 6. What Phase 0 Answers

Phase 0 answers the four intended acceptance questions:

1. The current Step7 final model has `26` errors: `13` false positives and
   `13` false negatives.
2. About one third of the residual errors are near the decision boundary, but
   many are confidently wrong.
3. The existing prediction-family ceiling is about `0.9267` for a single file
   and `0.9333` for simple average ensembles.
4. The remaining hard cases require new data, domain-aware routing, nonlinear
   stacking, or a third model branch.

## 7. Implications For Phase 1

Phase 1 should not collect generic extra data. It should target the residual
error buckets shown by the ledger:

1. Human hard negatives for poetry-like text:
   `poetry_freeverse`, `poetry_classical`, and short lyrical fragments.
2. Human hard negatives for polished or unusual prose:
   high-style literary passages and short reflective prose.
3. LLM hard positives for conservative rewrites:
   old-fiction style, poetry-preserving rewrites, and natural academic
   paraphrases.
4. A teacher-like development set with explicit `round2_tag`, `domain`,
   `subdomain`, `generator`, and `pair_id` metadata.

The next concrete step is to build `round2_teacher_like_dev.jsonl` and
`round2_teacher_like_train.jsonl` without using teacher-test samples or
teacher-test near duplicates.

## 8. Validation Performed

Checks run after implementation:

```powershell
.\.venv\Scripts\python.exe -m compileall src
git diff --check
```

Both checks passed.

The generated `outputs/round2/` artifacts are intentionally local outputs and
are ignored by Git under the existing `outputs/*` rule.
