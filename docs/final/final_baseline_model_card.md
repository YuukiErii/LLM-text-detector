# Final Baseline Model Card

Updated: 2026-05-22

## Status

The final baseline for this repository is the Step7 DeBERTa + TF-IDF ensemble.
It is the only neural checkpoint retained for public reproduction.

```text
DeBERTa branch: outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF branch:  outputs/models/tfidf_lit_academic_poetry
fusion config:  outputs/models/ensemble_lit_academic_poetry_step7_deberta_raw_tfidf/fusion_config.json
alpha:          0.5
threshold:      0.55
```

## Metrics

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

## Retained Files

The retained DeBERTa directory is pruned to inference artifacts:

```text
outputs/models/deberta_lit_academic_poetry_step7_combined/best_model/
outputs/models/deberta_lit_academic_poetry_step7_combined/tokenizer/
outputs/models/deberta_lit_academic_poetry_step7_combined/metrics.json
outputs/models/deberta_lit_academic_poetry_step7_combined/predictions/
```

The training checkpoints and optimizer states were removed from the repository
surface. The large `model.safetensors` file is tracked through Git LFS.

## Promotion Decision

Step7 remains final because later candidates did not beat it under strict
non-teacher promotion gates:

| Round | Candidate family | Result |
| --- | --- | --- |
| Round2 | bucket router, stacker, RoBERTa | hard-dev improvements did not transfer; teacher-test regressed or tied below Step7 |
| Round3 | ELECTRA, OOF stacker, precision guard | local repair was promising but teacher-test did not beat Step7 |
| Round4 | residual DeBERTa and human-style guard | useful hard-positive signal, unsafe human false positives |
| Round5 | flip guard residual override | passed non-teacher safety but made zero teacher-test overrides |
| Round6 | safe override selector | did not clear the fixed-FN requirement before teacher-test |
| Round7 | exact candidate selector | improved non-teacher unsafe blocking but tied Step7 at 274 / 300 |
| Round8 | residual-aware one-shot route | gate did not justify replacing Step7 |

The final report should frame the later rounds as diagnostic and reusable
training infrastructure, not as promoted models.

## Reproduction Command

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
