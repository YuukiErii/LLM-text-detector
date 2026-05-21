# Round3 Phase A-C Progress

Updated: 2026-05-21

This note records the first Round3 work block. It preserves the previous
Round2 artifacts and adds new Round3 outputs without overwriting Round2 files.

## 1. Phase A: Error-Delta Audit

New script:

```text
src/evaluation/round3_error_delta_audit.py
```

Generated outputs:

```text
outputs/round3/error_delta_audit.csv
outputs/round3/error_delta_audit.md
outputs/round3/error_delta_by_bucket.json
```

Key diagnostic finding:

| Candidate | Fixed Step7 errors | Fixed FN | Broke Step7 correct | Induced FP |
| --- | ---: | ---: | ---: | ---: |
| bucket_routed | 1 | 1 | 5 | 5 |
| stacker_step7 | 1 | 1 | 2 | 2 |
| roberta_single | 8 | 2 | 34 | 15 |
| stacker_with_roberta | 4 | 3 | 12 | 12 |

Only three Step7 false-negative IDs were fixed by any Round2 candidate:

```text
106, 107, 249
```

Candidate-induced human false positives appeared in high-risk text styles:
`poetry_classical`, `literary_short_fragment`, `academic_formal`, and polished
general prose. This supports the Round3 premise: use Step7 as the default and
only allow strongly guarded overrides.

## 2. Phase B: Precision-Guard Data

New script:

```text
src/data/build_round3_precision_guard_set.py
```

Generated outputs:

```text
data/processed/round3_hard_negative_mirror_source.jsonl
data/processed/round3_llm_hardpos_multi_generator_seed.jsonl
data/processed/round3_precision_guard_train.jsonl
data/processed/round3_precision_guard_dev.jsonl
data/processed/round3_precision_guard_spotcheck.jsonl
data/processed/round3_precision_guard_report.json
```

Accepted parts:

| Check | Result |
| --- | ---: |
| hard human negative pool | 9,762 |
| hard LLM positive pool | 2,872 |
| round3 precision-dev rows | 564 |
| precision-dev class balance | 282 human / 282 LLM |
| manual spotcheck packet | 80 rows |

Known data gap:

```text
human_literary_old_prose_mirror is still below the ideal 400-600 target.
```

The precision-dev set has paired coverage for most high-risk buckets, but
`literary_old_prose` still lacks enough local human/LLM mirror coverage for a
strong per-bucket conclusion.

## 3. Phase C: Balanced ELECTRA Branch

New training entrypoint:

```text
src/models/train_weighted_transformer.py
```

It supports:

```text
--sample_weight_field
--class_weight
--domain_weight_json
--balanced_sampler
```

Training run:

```text
model_name = google/electra-base-discriminator
train = data/processed/round3_precision_guard_train.jsonl
valid = data/processed/lit_academic_poetry_valid.jsonl
test = data/processed/lit_academic_poetry_internal_test.jsonl
guard_dev = data/processed/round3_precision_guard_dev.jsonl
epochs = 3
```

Generated outputs:

```text
outputs/models/round3_electra_base/
outputs/predictions/round3_electra_valid_predictions.jsonl
outputs/predictions/round3_electra_internal_test_predictions.jsonl
outputs/predictions/round3_electra_precision_guard_dev_predictions.jsonl
outputs/evaluation/round3_electra_report.md
```

Final ELECTRA metrics:

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9363 | 0.9554 | 0.9125 | 0.9335 | 36 | 74 |
| internal_test | 0.9341 | 0.9587 | 0.9044 | 0.9307 | 33 | 81 |
| round3_precision_guard_dev | 0.6525 | 0.8258 | 0.3865 | 0.5266 | 23 | 173 |

Step7 on the same precision-dev:

```text
F1 = 0.6481
FP = 10
FN = 142
```

Decision:

```text
Round3 ELECTRA v1 does not pass the Phase C gate.
```

It should not be promoted as a decision branch because it underperforms Step7
on both internal-test and the precision-guard dev set. It also worsens both FP
and FN on guard-dev.

## 4. Next Recommended Move

Do not proceed directly to OOF stacking with this ELECTRA run as a final
decision signal.

Recommended next step:

1. Keep Step7 as the default system.
2. Use Phase A audit results to design precision-guard override rules, but only
   around the tiny fixed-FN region (`106`, `107`, `249`) and without using
   teacher-test labels for tuning.
3. Improve Phase B data before retraining a second third branch:
   add old-prose human mirrors and more non-ChatGPT hard positives.
4. Try a lighter stylometry / char n-gram branch, or retrain ELECTRA with a less
   aggressive balanced sampler and a validation objective that explicitly
   protects hard human negatives.
