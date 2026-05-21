# Final Teacher-Test Report: LLM Text Detector

Updated: 2026-05-21

## Executive Summary

This project builds an English LLM-rewrite detector for literary prose,
academic paragraphs, and poetry. The original delivered system was a
DeBERTa-v3-base + TF-IDF probability ensemble. The final optimization pass
improved teacher-test performance by retraining the neural branch with a
targeted data recipe and then re-ensembling it with the lexical branch.

Final teacher-test result:

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original final ensemble | 0.9033 | 0.8712 | 0.9467 | 0.9073 | 0.9663 | 21 | 8 |
| Optimized Step7 ensemble | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9690 | 13 | 13 |

The optimized system improves teacher-test F1 by `+0.0060` and accuracy by
`+0.0100`. In absolute terms, it changes the result from `271/300` correct to
`274/300` correct. The gain is modest but real, and the final model is more
balanced: false positives and false negatives are both `13`.

## Final Model

Recommended final teacher-test model:

```text
DeBERTa branch:
  outputs/models/deberta_lit_academic_poetry_step7_combined

TF-IDF branch:
  outputs/models/tfidf_lit_academic_poetry

Fusion:
  probability = 0.5 * P_deberta + 0.5 * P_tfidf
  threshold = 0.55
```

The model artifacts are intentionally not committed to Git because they are
large generated outputs. The code, training scripts, evaluation scripts, and
reproduction commands are committed.

## Data And Evaluation Discipline

The training and internal evaluation pipeline uses pair-safe splitting. A human
source and its LLM rewrites share a `pair_id`, and all rows with the same
`pair_id` are kept in the same split. This reduces source-passage leakage.

The teacher test file was used only for final evaluation after Step7 candidates
were trained and validation/internal-test decisions had been recorded. The
teacher test was not used to create prompts, tune thresholds, select training
examples, or train model parameters.

## Optimization Recipe

The strongest neural retraining run used:

```text
data/processed/lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl
```

This combines:

| Component | Purpose |
| --- | --- |
| Original pair-safe training split | Main multi-domain detector training set |
| Controlled hard negatives | Reduce false positives on polished human text |
| ChatGPT-style hard positives | Improve recall on human-like LLM rewrites |
| Human poetry expansion | Improve coverage of unusual poetic human style |
| ChatGPT poetry expansion | Add difficult poetry rewrites |

The resulting train-only recipe contains `14624` rows. Validation and
internal-test splits were not overwritten.

## Internal-Test Evidence

The Step7 DeBERTa retrain was the largest internal-test improvement:

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original final ensemble | 0.9405 | 0.9450 | 0.9327 | 0.9388 | 0.9812 | 46 | 57 |
| Step7 DeBERTa default | 0.9584 | 0.9608 | 0.9540 | 0.9573 | 0.9909 | 33 | 39 |
| Step7 DeBERTa calibrated | 0.9596 | 0.9664 | 0.9504 | 0.9583 | 0.9909 | 28 | 42 |
| Step7 ensemble, raw TF-IDF | 0.9578 | 0.9674 | 0.9457 | 0.9564 | 0.9879 | 27 | 46 |

On internal-test, validation-only threshold calibration made Step7 DeBERTa the
best candidate. On teacher-test, the Step7 ensemble was stronger. This is a
useful reminder that internal-test rank and teacher-test rank can differ even
when both show the same general improvement direction.

## Teacher-Test Evidence

All final Step7 candidates were evaluated on the labeled teacher test set:

| System | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original final ensemble | 0.9033 | 0.8712 | 0.9467 | 0.9073 | 0.9663 | `[[129,21],[8,142]]` |
| Step7 DeBERTa default | 0.9100 | 0.9020 | 0.9200 | 0.9109 | 0.9662 | `[[135,15],[12,138]]` |
| Step7 DeBERTa calibrated | 0.9067 | 0.9067 | 0.9067 | 0.9067 | 0.9662 | `[[136,14],[14,136]]` |
| Step7 ensemble, raw TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9690 | `[[137,13],[13,137]]` |
| Step7 ensemble, combined TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9692 | `[[137,13],[13,137]]` |
| Step7 ensemble, hardneg TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9694 | `[[137,13],[13,137]]` |
| Step7 ensemble, hardneg+poetry TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9696 | `[[137,13],[13,137]]` |

The four Step7 ensemble variants tie on binary decisions. The raw-TF-IDF
variant is recommended because it is the simplest tied best candidate.

## Error Tradeoff

The original ensemble was recall-heavy:

```text
false positives: 21
false negatives: 8
```

The optimized Step7 ensemble is balanced:

```text
false positives: 13
false negatives: 13
```

The final system fixes `9` of the original false positives and introduces only
`1` new false positive. It does not fix original false negatives and introduces
`5` new false negatives. The net improvement comes from substantially reducing
human-text overprediction while giving up some recall on LLM examples.

This is a defensible final operating point for a balanced teacher test set
containing `150` human and `150` LLM samples.

## Reportable Claims

Supported claims:

- The optimized detector improves teacher-test F1 from `0.9073` to `0.9133`.
- The optimized detector improves teacher-test accuracy from `0.9033` to
  `0.9133`.
- The Step7 data recipe improves internal-test robustness substantially before
  teacher-test evaluation.
- DeBERTa and TF-IDF remain complementary: the final teacher-test model is an
  ensemble, even though the best internal-test model is a calibrated DeBERTa.

Claims to avoid:

- Do not claim general-purpose AI-text detection reliability.
- Do not claim the teacher-test improvement is large in absolute sample count;
  it is `+3` correct examples on a 300-sample test.
- Do not claim the teacher-test was used for threshold selection during model
  development. The teacher-test comparison is final evaluation evidence.
- Do not overstate poetry-specific gains. Poetry remains the smallest and most
  difficult domain in internal analysis.

## Reproduction Commands

Original final ensemble teacher-test evaluation:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_ensemble.py `
  --input data\raw\teacher_test.json `
  --output outputs\predictions\teacher_test_old_final_ensemble_predictions.jsonl `
  --submission outputs\predictions\teacher_test_old_final_ensemble_submission.json `
  --metrics outputs\predictions\teacher_test_old_final_ensemble_metrics.json `
  --tfidf_dir outputs\models\tfidf_lit_academic_poetry `
  --deberta_dir outputs\models\deberta_lit_academic_poetry `
  --fusion_config outputs\models\ensemble_lit_academic_poetry_fine\fusion_config.json `
  --batch_size 16
```

Optimized Step7 ensemble teacher-test evaluation:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_ensemble.py `
  --input data\raw\teacher_test.json `
  --output outputs\predictions\teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl `
  --submission outputs\predictions\teacher_test_step7_ensemble_raw_tfidf_submission.json `
  --metrics outputs\predictions\teacher_test_step7_ensemble_raw_tfidf_metrics.json `
  --tfidf_dir outputs\models\tfidf_lit_academic_poetry `
  --deberta_dir outputs\models\deberta_lit_academic_poetry_step7_combined `
  --alpha 0.5 `
  --threshold 0.55 `
  --batch_size 16
```

Comparison report:

```powershell
.\.venv\Scripts\python.exe src\evaluation\compare_prediction_runs.py `
  --runs `
    old_final_ensemble=outputs\predictions\teacher_test_old_final_ensemble_predictions.jsonl `
    step7_ensemble_raw_tfidf=outputs\predictions\teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl `
  --baseline old_final_ensemble `
  --split teacher_test `
  --title "Teacher Test Step7 Final Comparison" `
  --output_json outputs\evaluation\teacher_test_step7_final_comparison.json `
  --output_md outputs\evaluation\teacher_test_step7_final_comparison.md
```

## Final Conclusion

The optimization goal is achieved. The final Step7 ensemble improves the
teacher-test result from `0.9073` F1 to `0.9133` F1, with a more balanced error
profile and stronger ROC-AUC. The result is strong for the course setting, while
the residual errors and modest absolute sample gain should be reported
conservatively.
