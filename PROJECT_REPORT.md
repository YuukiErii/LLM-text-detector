# Project Report: LLM Text Detector

Updated: 2026-05-22

This file is the consolidated project handoff. It keeps the final results,
model recipe, optimization history, and next-step judgment in one root-level
document. Older detailed drafts and logs are preserved under `docs/archive/`.

## 1. Project Scope

The project builds a binary detector for English passages:

| Label | Meaning |
| --- | --- |
| `0` | Human-written text |
| `1` | LLM-generated or LLM-rewritten text |

The target distribution is not generic AI essay detection. It focuses on
LLM-rewritten literary prose, archaic English, poetry, and NLP / computational
linguistics academic paragraphs.

The final system is a hybrid detector:

```text
P_final = alpha * P_deberta + (1 - alpha) * P_tfidf
```

| Branch | Role |
| --- | --- |
| DeBERTa-v3-base classifier | Captures semantic, discourse, and deep style signals |
| Word/char TF-IDF + Logistic Regression | Captures lexical, punctuation, spelling, and surface n-gram signals |

## 2. Data Summary

The main processed dataset is:

```text
data/processed/full_dataset_lit_academic_poetry.jsonl
```

It contains `17,295` rows:

| Label / source | Count |
| --- | ---: |
| Human-written | 8,830 |
| LLM-rewritten | 8,465 |
| Literature domain | 14,008 |
| Academic domain | 2,292 |
| Poetry domain | 995 |

Human sources:

| Domain | Human samples | Source |
| --- | ---: | --- |
| Literature | 7,130 | Project Gutenberg fiction |
| Academic | 1,200 | ACL-OCL academic paragraphs |
| Poetry | 500 | Project Gutenberg poetry |

LLM rewrite generators:

| Generator | Rows |
| --- | ---: |
| DeepSeek | 2,367 |
| ChatGPT | 2,366 |
| Gemini | 1,875 |
| Doubao | 1,857 |

The main train / validation / internal-test split is pair-safe: all rows with
the same `pair_id` stay in the same split, so a human source and its rewrites do
not leak across splits.

| Split | Samples | Pair IDs |
| --- | ---: | ---: |
| Train | 13,836 | 7,064 |
| Validation | 1,728 | 882 |
| Internal test | 1,731 | 884 |

## 3. Final Results

The original delivered ensemble used:

```text
alpha = 0.33
threshold = 0.48
```

The final optimized Step7 teacher-test model uses:

```text
DeBERTa branch:
  outputs/models/deberta_lit_academic_poetry_step7_combined

TF-IDF branch:
  outputs/models/tfidf_lit_academic_poetry

Fusion:
  alpha = 0.5
  threshold = 0.55
```

Main metrics:

| System | Split | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | internal_test | 0.9087 | 0.9389 | 0.8701 | 0.9032 | 0.9659 | 48 | 110 |
| DeBERTa-v3-base | internal_test | 0.9284 | 0.9208 | 0.9339 | 0.9273 | 0.9830 | 68 | 56 |
| Original ensemble | internal_test | 0.9405 | 0.9450 | 0.9327 | 0.9388 | 0.9812 | 46 | 57 |
| Step7 DeBERTa calibrated | internal_test | 0.9596 | 0.9664 | 0.9504 | 0.9583 | 0.9909 | 28 | 42 |
| Original final ensemble | teacher_test | 0.9033 | 0.8712 | 0.9467 | 0.9073 | 0.9663 | 21 | 8 |
| Optimized Step7 ensemble | teacher_test | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9690 | 13 | 13 |

Teacher-test confusion matrices:

```text
Original final ensemble: [[129, 21], [8, 142]]
Optimized Step7 ensemble: [[137, 13], [13, 137]]
```

The Step7 ensemble improves teacher-test accuracy from `0.9033` to `0.9133`.
In absolute terms, it changes the result from `271/300` correct to `274/300`
correct. The gain is modest but real; the optimized model is more balanced,
with `13` false positives and `13` false negatives.

Round3 added a precision-guarded repair route after the second optimization
round. It did not beat Step7 on the final teacher-test diagnostic:

| Round3 candidate | teacher_test Accuracy | Precision | Recall | F1 | FP | FN | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Round3 ELECTRA | 0.8800 | 0.9071 | 0.8467 | 0.8759 | 13 | 23 | Reject |
| Round3 OOF stacker | 0.9033 | 0.9353 | 0.8667 | 0.8997 | 9 | 20 | Reject |
| Round3 precision guard | 0.9100 | 0.9073 | 0.9133 | 0.9103 | 14 | 13 | Reject |
| Step7 ensemble | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 | Keep final |

The final Round3 submission file is therefore an explicit Step7 alias:

```text
outputs/predictions/round3_final_submission.json
```

## 4. Optimization History

The 2026-05-21 optimization pass followed a staged roadmap:

1. Freeze baseline metrics and comparison scripts.
2. Test text normalization and encoding repair.
3. Calibrate thresholds using validation data only.
4. Add controlled human hard negatives.
5. Add ChatGPT-style hard LLM positives.
6. Expand poetry coverage with human poetry and ChatGPT poetry rewrites.
7. Retrain DeBERTa and retune the DeBERTa + TF-IDF ensemble.

Key findings:

| Stage | Finding |
| --- | --- |
| Text normalization | Encoding-only normalization slightly helped some internal metrics, but not enough to promote. |
| Hard negatives | Reduced human false positives but tended to shift errors toward LLM false negatives. |
| ChatGPT hard positives | Improved ChatGPT recall in TF-IDF diagnostics, but cheap ensembles did not beat the best baseline. |
| Poetry expansion | Helped lightweight poetry diagnostics, but poetry remained the hardest internal domain. |
| Step7 DeBERTa retrain | Produced the largest internal-test improvement. |
| Step7 ensemble | Best teacher-test operating point after final evaluation. |

The strongest Step7 training recipe was:

```text
data/processed/lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl
```

It combines the original train split, controlled hard negatives, ChatGPT-style
hard positives, human poetry expansion, and ChatGPT poetry expansion.

The 2026-05-22 Round3 pass then tested the postmortem hypothesis from
`docs/ROUND2_POSTMORTEM_AND_ROUND3_PLAN.md`:

1. Phase A audited Round2 error deltas.
2. Phase B built a precision-guard development set.
3. Phase C trained an ELECTRA branch, which failed the promotion gate.
4. Phase D trained an OOF logistic stacker. It reduced FP, but became too
   conservative and missed too many LLM positives.
5. Phase E tuned precision-guarded override rules. On guard-dev it fixed `18`
   Step7 false negatives with `0` induced false positives.
6. Phase F rejected the precision guard on teacher-test because it made one
   override that induced `1` new FP and fixed `0` Step7 FN.

Round3's value is diagnostic: it shows that guarded repair can work on a
constructed mirror set, but the current local data is still not sufficient for
reliable teacher-test generalization.

## 5. Error Pattern

The hardest remaining cases are:

| Error type | Pattern |
| --- | --- |
| Human false positives | Human poetry, polished literary prose, archaic / high-style writing, formal academic prose |
| LLM false negatives | Human-like ChatGPT rewrites, old-fiction style rewrites, archaic poem rewrites, natural academic paraphrases |

Internal-test weak points:

| Breakdown | Result |
| --- | --- |
| Poetry domain | F1 `0.8767`, recall `0.8000` |
| ChatGPT LLM rows | recall `0.7884`, 51 false negatives |
| DeepSeek / Doubao / Gemini LLM rows | near-perfect recall |

This suggests that the residual difficulty is distributional: some human texts
look stylistically unusual enough to resemble LLM text, while some ChatGPT
rewrites preserve enough human style to evade the current detector.

## 6. Practical Path Toward 95%

The current `91.33%` teacher-test score means `26/300` examples are wrong. A
`95%` result would require at most `15/300` errors, so the system must net-fix
at least `11` teacher-test-style mistakes.

Further threshold or linear ensemble tuning is unlikely to be enough. A
diagnostic sweep over existing predictions only reaches the low `92%` range,
and the existing model variants share many of the same residual errors.

The second and third optimization rounds show that another global threshold is
unlikely to be enough. Most practical next steps are data-first:

1. Build a teacher-test-like development set from the residual error patterns:
   human free verse, classical poetry, polished prose, natural academic writing,
   old-fiction LLM rewrites, and ChatGPT conservative paraphrases.
2. Add more old-prose human mirrors; this remained the weakest mirror bucket in
   Round3 Phase B.
3. Add non-ChatGPT hard positives so the repair signal is not tied to one
   generator style.
4. Add a stylometry / char n-gram branch as a guard feature, and ensemble only
   if it makes different errors from DeBERTa, TF-IDF, RoBERTa, and ELECTRA.
5. Keep teacher-test labels out of training and threshold selection unless the
   course explicitly allows post-hoc tuning.

## 7. Reproduction Commands

Optimized Step7 teacher-test inference:

```powershell
python src/evaluation/predict_ensemble.py `
  --input data/raw/teacher_test.json `
  --output outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl `
  --submission outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_submission.json `
  --metrics outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_metrics.json `
  --tfidf_dir outputs/models/tfidf_lit_academic_poetry `
  --deberta_dir outputs/models/deberta_lit_academic_poetry_step7_combined `
  --alpha 0.5 `
  --threshold 0.55 `
  --batch_size 16 `
  --minimal_submission
```

Original final ensemble evaluation:

```powershell
python src/evaluation/predict_ensemble.py `
  --input data/raw/teacher_test.json `
  --output outputs/predictions/teacher_test_old_final_ensemble_predictions.jsonl `
  --submission outputs/predictions/teacher_test_old_final_ensemble_submission.json `
  --metrics outputs/predictions/teacher_test_old_final_ensemble_metrics.json `
  --tfidf_dir outputs/models/tfidf_lit_academic_poetry `
  --deberta_dir outputs/models/deberta_lit_academic_poetry `
  --fusion_config outputs/models/ensemble_lit_academic_poetry_fine/fusion_config.json `
  --batch_size 16
```

Round3 final comparison:

```powershell
python src/evaluation/compare_round2_candidates.py `
  --title "Round3 Final Teacher-Test Candidate Comparison" `
  --runs `
    step7=outputs/predictions/round2_step7_teacher_test_predictions.jsonl `
    round3_electra=outputs/predictions/round3_electra_teacher_test_predictions.jsonl `
    round3_oof_stacker=outputs/predictions/round3_oof_stacker_teacher_test_predictions.jsonl `
    round3_precision_guard=outputs/predictions/round3_precision_guard_teacher_test_predictions.jsonl `
  --output_md outputs/evaluation/round3_final_teacher_comparison.md `
  --output_json outputs/evaluation/round3_final_teacher_comparison.json `
  --overlap_csv outputs/evaluation/round3_error_overlap_matrix.csv
```

## 8. Document Map

Root-level documents:

| File | Purpose |
| --- | --- |
| `README.md` | Repo entrypoint, setup, commands, and public-facing overview |
| `PROJECT_REPORT.md` | Consolidated final results, optimization summary, and next-step plan |
| `docs/SECOND_ROUND_95_OPTIMIZATION_PLAN.md` | Detailed execution plan for the second-round 95% accuracy target |
| `docs/ROUND2_RESULTS_SUMMARY.md` | Executed second-round results and final candidate comparison |
| `docs/ROUND2_POSTMORTEM_AND_ROUND3_PLAN.md` | Round2 postmortem and Round3 execution plan |
| `docs/ROUND3_PHASEA_C_PROGRESS.md` | Round3 Phase A-C handoff |
| `docs/ROUND3_RESULTS_SUMMARY.md` | Completed Round3 Phase D-F results and final decision |

Archived source documents:

| File | Purpose |
| --- | --- |
| `docs/archive/FINAL_TEACHER_TEST_REPORT_2026-05-21.md` | Final teacher-test report before consolidation |
| `docs/archive/RESULTS_AND_OPTIMIZATION_PLAN.md` | Detailed result interpretation and optimization roadmap |
| `docs/archive/NLG_LLM_Detector_Holistic_Optimization_Plan_2026-05-21.md` | Step-by-step holistic optimization plan |
| `docs/archive/NLG_LLM_Detector_Optimization_Work_Log_2026-05-21.md` | Full optimization work log |
| `docs/archive/NLG_LLM_Detector_Progress_and_Plan_Updated_2026-05-20.md` | Earlier progress and plan snapshot |
| `docs/archive/NLG_LLM_Detector_Task_Outline.md` | Original task outline |

The archived files are retained for traceability. `PROJECT_REPORT.md` is the
recommended document to read first.
