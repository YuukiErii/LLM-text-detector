# Round5 Final Decision

Date: 2026-05-22

This file records the Phase 7 teacher-test diagnostic for Round5. The selected
Round5 override rule was frozen before teacher-test and was applied without
teacher-test threshold tuning or rule tuning.

## Final Result

```text
PROMOTE_TO_TEACHER_TEST = completed
FINAL_MODEL_CANDIDATE = no
PROMOTE_AS_FINAL = no
KEEP_FINAL_MODEL = Step7 ensemble
```

Round5 does not beat Step7 on teacher-test:

| Run | Correct / 300 | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step7 baseline | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |
| Round5 override | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |

Target status:

```text
beat Step7 target: not met
95% target: not met
needed for 95%: 285 / 300
current Round5: 274 / 300
```

## Frozen Rule Applied

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

## Teacher-Test Diagnostic

Intermediate teacher-test signal branch:

| Branch | Correct / 300 | Accuracy | FP | FN | Meaning |
| --- | ---: | ---: | ---: | ---: | --- |
| Step7 | 274 | 0.9133 | 13 | 13 | current final baseline |
| Round4 DeBERTa branch | 263 | 0.8767 | 23 | 14 | useful as signal, unsafe as global classifier |
| Round5 override | 274 | 0.9133 | 13 | 13 | no-op on teacher-test |

Round4-vs-Step7 teacher-test flip ledger:

| Flip type | Count |
| --- | ---: |
| stable_step7_correct | 256 |
| both_miss | 19 |
| induced_fp | 14 |
| round4_induced_fn | 4 |
| round4_fixed_fp | 4 |
| fixed_fn_candidate | 3 |

Override candidates:

| Type | Count |
| --- | ---: |
| unsafe induced FP | 14 |
| safe fixed FN candidate | 3 |
| total Step7-human -> Round4-LLM candidates | 17 |

Why Round5 made zero teacher-test overrides:

1. The frozen bucket rule only allows `literary_old_prose` and
   `literary_short_fragment`.
2. One safe candidate was `general_prose`, so it was not eligible.
3. The two safe `literary_short_fragment` candidates had
   `p_unsafe_override > 0.35`, so the flip guard vetoed them.
4. Several induced-FP candidates were correctly blocked by the same constraints.

This is a conservative failure: Round5 avoided damaging Step7 but did not repair
any teacher-test false negatives.

## Artifacts

```text
outputs/predictions/round5_step7_teacher_test_predictions.jsonl
outputs/predictions/round5_round4_deberta_teacher_test_predictions.jsonl
outputs/predictions/round5_human_style_guard_teacher_test_predictions.jsonl
outputs/predictions/round5_flip_guard_teacher_test_predictions.jsonl
outputs/predictions/round5_teacher_test_predictions.jsonl
outputs/evaluation/round5_teacher_test_ledger.jsonl
outputs/evaluation/round5_teacher_test_ledger_summary.json
outputs/evaluation/round5_teacher_test_comparison.md
outputs/evaluation/round5_teacher_test_comparison.json
```

## Next Recommendation

Do not promote Round5 as final. Keep Step7 as the final model.

The next optimization should focus on the three teacher-test-like safe override
patterns without using teacher-test for tuning. The practical route is:

1. Build non-teacher dev data that mirrors `general_prose` and
   `literary_short_fragment` safe fixed-FN cases.
2. Improve flip-guard calibration so it can distinguish safe short-fragment LLM
   rewrites from human short-fragment induced FP.
3. Keep the hard rule that hardneg induced FP must remain zero before another
   teacher-test diagnostic.
