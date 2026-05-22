# Round7 Final Decision

Date: 2026-05-22

This file records the single frozen Round7 teacher-test diagnostic after the
non-teacher Round7 gate passed.

The teacher-test boundary remained intact:

```text
Round7 exact-candidate data mining, selector training, threshold selection, and
override rule search were completed before this diagnostic.
The teacher-test run only scored the frozen selector and applied the frozen rule.
```

## Final Result

```text
PROMOTE_TO_TEACHER_TEST = completed
FINAL_MODEL_CANDIDATE = no
PROMOTE_AS_FINAL = no
KEEP_FINAL_MODEL = Step7 ensemble
```

Round7 does not beat Step7 on teacher-test:

| Run | Correct / 300 | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step7 baseline | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |
| Round4 signal branch | 263 | 0.8767 | 0.8553 | 0.9067 | 0.8803 | 23 | 14 |
| Round7 exact override | 274 | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |

Target status:

```text
beat Step7 target: not met
95% target: not met
needed for 95%: 285 / 300
current Round7: 274 / 300
```

## Frozen Rule Applied

```json
{
  "p_round7_safe_min": 0.45,
  "p_unsafe_max": 0.35,
  "human_style_max": 1.0,
  "round4_threshold": 0.5,
  "min_delta": 0.0,
  "bucket_policy": "old_short_plus_general_strict",
  "general_p_round7_safe_min": 0.55,
  "general_p_unsafe_max": 0.25,
  "general_round4_threshold": 0.55,
  "general_min_delta": 0.05,
  "disabled_baseline": false
}
```

## Teacher-Test Diagnostic

The Round7 selector was scored on the already-aligned teacher-test flip-guard
ledger, then the frozen rule was applied once.

Teacher-test exact override surface:

| Candidate type | Count |
| --- | ---: |
| safe fixed-FN candidate | 3 |
| unsafe induced-FP candidate | 14 |
| total Step7-human -> Round4-LLM candidates | 17 |

Round7 override delta:

| Item | Count |
| --- | ---: |
| overrides | 0 |
| fixed Step7 FN | 0 |
| induced FP | 0 |
| broke Step7 correct | 0 |

Why Round7 made zero teacher-test overrides:

1. The safe candidate surface is still only three rows against fourteen unsafe
   Round4-induced FP candidates.
2. The one safe `general_prose` candidate fails the frozen general strict
   `p_unsafe_override <= 0.25` guard.
3. The two safe `literary_short_fragment` candidates fail the frozen base
   `p_unsafe_override <= 0.35` guard.
4. That same conservative guard also prevents the unsafe candidates from
   becoming Round7 teacher-test false positives.

Interpretation:

```text
Round7 improved the non-teacher exact-candidate safety surface substantially,
but the final teacher-test safe candidates still look unsafe to the frozen
Round5 unsafe guard. Round7 is therefore a safe no-op on teacher-test.
```

## Artifacts

```text
outputs/predictions/round7_exact_selector_teacher_test_predictions.jsonl
outputs/predictions/round7_teacher_test_predictions.jsonl
outputs/evaluation/round7_teacher_test_comparison.json
outputs/evaluation/round7_teacher_test_comparison.md
outputs/evaluation/round7_teacher_test_ledger_summary.json
```

## Next Recommendation

Keep Step7 as the final model.

The next optimization should not loosen the frozen Round7 teacher-test rule. The
useful next route is to revisit the unsafe-guard bottleneck on non-teacher data:

1. Build exact-candidate training/dev coverage that separates safe
   `general_prose` and safe `literary_short_fragment` from the current
   high-style human unsafe candidates.
2. Decide whether Round5 `p_unsafe_override` should remain a hard veto or become
   a calibrated Round8 feature under a new non-teacher gate.
3. Preserve the requirement that hardneg induced FP stays zero before any later
   teacher-test diagnostic.
