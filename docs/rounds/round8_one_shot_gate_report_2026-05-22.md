# Round8 One-Shot Gate Report

This report uses non-teacher surfaces only. Teacher-test diagnostics remain blocked unless all gates pass.

## Metric Deltas

| Split | Step7 F1 | Round8 F1 | F1 Delta | Net Correct | New FP | Fixed FN | Used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9564 | 0.9570 | 0.0006 | 1 | 4 | 7 | 45 |
| residual_dev | 0.2060 | 0.2363 | 0.0303 | 8 | 0 | 10 | 24 |
| residual_probe | 0.1723 | 0.2406 | 0.0683 | 19 | 3 | 9 | 51 |

## Gates

| Gate | Required | Observed | Pass |
| --- | --- | --- | --- |
| internal_test_f1_retention | >= Step7 - 0.0030 | 0.9570 vs Step7 0.9564 | True |
| residual_dev_f1_gain | >= Step7 + 0.0400 | delta 0.0303 | False |
| residual_probe_f1_gain | >= Step7 + 0.0300 | delta 0.0683 | True |
| internal_test_new_fp | <= 1 | 4 | False |
| residual_probe_new_fp | <= 1 | 3 | False |
| residual_probe_net_correct | >= 0 | 19 | True |

## Confidence Sensitivity

The trained selector selected confidence threshold `0.63` on the dedicated
ambiguous dev split. The original plan's simple threshold `0.65` was also
checked. A wider threshold sweep showed no confidence threshold satisfying all
strict non-teacher gates at once.

| Confidence | internal F1 delta | internal new FP | residual_dev F1 delta | residual_probe F1 delta | residual_probe new FP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.0030 | 4 | 0.0414 | 0.0665 | 7 |
| 0.63 | 0.0006 | 4 | 0.0303 | 0.0683 | 3 |
| 0.65 | 0.0012 | 4 | 0.0303 | 0.0683 | 3 |
| 0.70 | 0.0024 | 3 | 0.0303 | 0.0692 | 1 |
| 0.75 | 0.0031 | 3 | 0.0265 | 0.0626 | 0 |
| 0.95 | 0.0006 | 1 | 0.0115 | 0.0351 | 0 |

## Decision

```text
ROUND8_TEACHER_TEST_DIAGNOSTIC_ALLOWED = no; keep Step7 as final baseline and treat the selector as diagnostic/reusable only.
```
