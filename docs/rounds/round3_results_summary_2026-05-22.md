# Round3 Results Summary

Updated: 2026-05-22

This note records the completed third optimization round. The source strategy
was `docs/rounds/round2_postmortem_and_round3_plan_2026-05-22.md`; the Phase A-C handoff was
`docs/rounds/round3_phase_a_to_c_progress_2026-05-21.md`.

## 1. Final Decision

Round3 did not produce a candidate that beats the strict Step7 baseline on the
final teacher-test diagnostic.

The final system remains:

```text
DeBERTa: outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:  outputs/models/tfidf_lit_academic_poetry
alpha:   0.5
threshold: 0.55
```

Final submission alias:

```text
outputs/predictions/round3_final_submission.json
```

This file intentionally matches the Step7 teacher-test submission. The Round3
precision-guarded candidate improved the non-teacher guard sets, but it did not
improve the teacher-test result.

## 2. Phase D: OOF Stacker

New scripts:

```text
src/models/train_round3_oof_stacker.py
src/evaluation/predict_round3_oof_stacker.py
src/evaluation/round3_fusion_utils.py
```

Base signals:

```text
step7
round2_roberta
round3_electra
```

Meta-training splits:

```text
valid
round2_teacher_like_dev
round3_precision_guard_dev
```

OOF threshold selection used only non-teacher-test splits. The selected global
OOF stacker threshold was:

```text
threshold = 0.87
```

Main result:

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OOF meta train | 0.8654 | 0.9673 | 0.7364 | 0.8362 | 39 | 413 |
| internal_test | 0.9440 | 0.9847 | 0.8996 | 0.9402 | 12 | 85 |
| teacher_test | 0.9033 | 0.9353 | 0.8667 | 0.8997 | 9 | 20 |

Decision:

```text
The OOF stacker is too conservative as a global classifier.
It lowers false positives, but misses too many LLM positives.
It should not replace Step7.
```

Related outputs:

```text
outputs/models/round3_oof_stacker/
outputs/evaluation/round3_oof_stacker_report.md
outputs/predictions/round3_oof_stacker_internal_test_predictions.jsonl
outputs/predictions/round3_oof_stacker_teacher_test_predictions.jsonl
```

## 3. Phase E: Precision-Guarded Routing

New scripts:

```text
src/evaluation/tune_precision_guard_rules.py
src/evaluation/predict_precision_guarded_ensemble.py
```

Selected rule:

```text
default prediction = Step7

Allow human -> LLM override only when:
  p_oof >= 0.85
  p_roberta >= 0.65
  min_votes >= 2
  min_words >= 32

ELECTRA was not required by the selected rule.
```

On non-teacher tuning splits, this rule looked useful:

| Split | Accuracy | Precision | Recall | F1 | FP | FN | Step7 FP | Step7 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.9659 | 0.9668 | 0.9634 | 0.9651 | 28 | 31 | 28 | 33 |
| round2_dev | 0.7822 | 0.8966 | 0.5330 | 0.6686 | 27 | 205 | 24 | 230 |
| guard_dev | 0.7624 | 0.9405 | 0.5603 | 0.7022 | 10 | 124 | 10 | 142 |

Guard-dev interpretation:

```text
fixed Step7 FN = 18
induced FP = 0
```

This was a genuine precision-guard improvement on the constructed guard set.
It was therefore allowed to advance to Phase F for diagnostic comparison.

Related outputs:

```text
outputs/models/round3_precision_guard/rules.json
outputs/evaluation/round3_precision_guard_tuning_report.md
outputs/predictions/round3_precision_guard_internal_test_predictions.jsonl
outputs/predictions/round3_precision_guard_teacher_test_predictions.jsonl
```

## 4. Phase F: Final Comparison

Teacher-test comparison:

| Candidate | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |
| round2_bucket_routed | 0.9000 | 0.8846 | 0.9200 | 0.9020 | 18 | 12 |
| round2_stacker | 0.9100 | 0.9013 | 0.9200 | 0.9109 | 15 | 12 |
| round2_roberta | 0.8267 | 0.8455 | 0.8000 | 0.8219 | 22 | 30 |
| round2_stacker_with_roberta | 0.8867 | 0.8537 | 0.9333 | 0.8917 | 24 | 10 |
| round3_electra | 0.8800 | 0.9071 | 0.8467 | 0.8759 | 13 | 23 |
| round3_oof_stacker | 0.9033 | 0.9353 | 0.8667 | 0.8997 | 9 | 20 |
| round3_precision_guard | 0.9100 | 0.9073 | 0.9133 | 0.9103 | 14 | 13 |

The final Round3 precision guard made one teacher-test override:

```text
overrides = 1
fixed Step7 FN = 0
induced FP = 1
```

Therefore it is rejected as the final system. The best strict candidate remains
Step7 with `274 / 300` correct.

Related outputs:

```text
outputs/evaluation/round3_internal_comparison.md
outputs/evaluation/round3_round2_dev_comparison.md
outputs/evaluation/round3_precision_guard_dev_comparison.md
outputs/evaluation/round3_final_teacher_comparison.md
outputs/evaluation/round3_error_overlap_matrix.csv
```

## 5. What Round3 Proved

Round3 was useful even though it did not reach 95%.

Confirmed:

1. The new ELECTRA branch is not a safe final decision signal under the current
   training recipe.
2. OOF stacking reduces overconfident false positives but becomes too
   conservative and loses LLM recall.
3. Precision-guarded overrides can improve constructed hard-dev and guard-dev
   sets without adding guard-dev false positives.
4. The teacher-test distribution still contains high-style human examples that
   make even carefully gated overrides risky.

The main remaining bottleneck is not another global threshold. It is the lack
of enough local data that simultaneously covers:

```text
hard LLM positives that Step7 misses
matched high-style human negatives that prevent new FP
```

## 6. Recommended Next Work

The next useful work is data-first:

1. Add more old-prose human mirrors; this bucket was still under-covered after
   Phase B.
2. Add non-ChatGPT hard positives so the repair signal is not tied to one
   generator style.
3. Build a small stylometry / char n-gram branch and evaluate it as a guard
   feature, not as a global classifier.
4. Keep teacher-test labels out of tuning. Use them only for final diagnostics.

If no new data is added, the safest public result remains the Step7 ensemble.
