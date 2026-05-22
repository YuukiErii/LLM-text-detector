# Three-Round Optimization Review And 95% Route

Updated: 2026-05-22

This note reviews the three completed optimization rounds for the LLM text
detector and analyzes the most plausible path toward 95% accuracy on the
teacher-held test set.

The central conclusion is:

```text
The current strict-route final model should remain the Step7 DeBERTa + TF-IDF
ensemble. Reaching 95% is unlikely to come from another global threshold,
another simple stacker, or a single new transformer branch. The most plausible
route is a data-first, error-bucket-specific repair loop: build paired hard LLM
positive and matched hard human negative data, then train a stronger calibrated
neural branch and use it only for precision-guarded local overrides.
```

## 1. Current Final Baseline

The current strict-route final model is:

```text
DeBERTa:  outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:   outputs/models/tfidf_lit_academic_poetry
alpha:    0.5
threshold: 0.55
```

Teacher-test result:

| System | Accuracy | Correct | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step7 ensemble | 0.9133 | 274 / 300 | 0.9133 | 0.9133 | 0.9133 | 13 | 13 |

95% teacher-test accuracy means:

```text
target correct = 285 / 300
current correct = 274 / 300
needed net gain = +11 correct examples
current errors = 26
maximum allowed errors at 95% = 15
```

This is a large remaining gap because the current errors are balanced. A method
that fixes a few false negatives but creates new false positives will not move
the final score much, and can easily make it worse.

## 2. Round 1: Full Pipeline And Step7 Optimization

Round 1 built the full detector pipeline:

- literature, academic, and poetry human seeds;
- LLM rewrites from ChatGPT, DeepSeek, Gemini, and Doubao;
- pair-safe train / validation / internal-test splits;
- TF-IDF logistic regression baseline;
- DeBERTa-v3-base classifier;
- DeBERTa + TF-IDF probability ensemble;
- error analysis and final teacher-test inference scripts.

The first final teacher-test system was:

```text
P_final = 0.33 * P_deberta + 0.67 * P_tfidf
threshold = 0.48
```

It reached:

| System | Accuracy | Correct | F1 | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original final ensemble | 0.9033 | 271 / 300 | 0.9073 | 21 | 8 | [[129, 21], [8, 142]] |

The model was recall-heavy: it caught most LLM texts but over-predicted LLM on
polished or unusual human writing.

Round 1 then tried a broad holistic optimization path:

1. hard-negative human data;
2. text normalization and encoding-only normalization;
3. validation-only calibration;
4. controlled hard-negative quotas;
5. ChatGPT-style hard positives;
6. poetry expansion;
7. neural retraining with the best data recipe.

The decisive improvement came from Step7 neural retraining:

```text
data/processed/lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl
```

Teacher-test comparison:

| System | Accuracy | Correct | F1 | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original final ensemble | 0.9033 | 271 / 300 | 0.9073 | 21 | 8 | [[129, 21], [8, 142]] |
| Step7 ensemble | 0.9133 | 274 / 300 | 0.9133 | 13 | 13 | [[137, 13], [13, 137]] |

Round 1's important lesson:

```text
Balanced targeted training data can move the real teacher-test result. Pure
threshold changes and TF-IDF-only augmentations are useful diagnostics, but the
main transferable gain came when the neural branch absorbed the hard-negative,
hard-positive, and poetry-expansion evidence.
```

## 3. Round 2: 95% Attempt With Teacher-Like Dev, Router, Stacker, RoBERTa

Round 2 started from an explicit 95% target:

```text
95% = 285 / 300 correct
Step7 = 274 / 300 correct
net gain needed = +11
```

It preserved the strict route:

```text
teacher-test labels were not used for training, threshold selection, router
tuning, stacker training, or model selection.
```

### 3.1 Phase 0: Existing-Family Ceiling

Round2 Phase 0 added:

```text
src/evaluation/export_error_ledger.py
src/evaluation/round2_threshold_family_diagnostics.py
```

Key findings:

| Diagnostic | Result |
| --- | ---: |
| Step7 residual errors | 26 = 13 FP + 13 FN |
| Near-boundary residual errors | 9 / 26 |
| Confidently wrong residual errors | 17 / 26 |
| Best single existing prediction file with oracle teacher-test threshold | about 0.9267 |
| Best simple average of existing prediction files with oracle teacher-test threshold | about 0.9333 |

This matters because even invalid diagnostic-only oracle thresholding could not
reach 95%. The current model family simply does not contain enough independent
signal.

### 3.2 Phase 1: Teacher-Like Development Data

Round2 built residual-error-oriented data:

```text
data/processed/round2_human_hardneg_seed.jsonl
data/processed/round2_llm_hardpos_seed.jsonl
data/processed/round2_teacher_like_train.jsonl
data/processed/round2_teacher_like_dev.jsonl
```

Accepted data checks:

| Check | Result |
| --- | ---: |
| hard buckets covered | 9 |
| round2 dev rows | 1065 |
| round2 train additions | 3027 |
| dev minimum class share | 41.2% |
| poetry represented | yes |
| academic represented | yes |

The new dev set was intentionally difficult. Step7 fell to:

```text
round2 teacher-like dev F1 = 0.6220
confusion = [[602, 24], [230, 209]]
```

This exposed the hard-positive problem, especially conservative ChatGPT-style
rewrites, old-fiction style, poetry-preserving rewrites, and natural academic
paraphrases.

### 3.3 Phase 2: Domain Router

Round2 added domain/bucket routing:

```text
poetry_classical
poetry_freeverse
literary_old_prose
literary_short_fragment
academic_formal
general_prose
```

Result:

| Split | Step7 F1 | Bucket-routed F1 | Main effect |
| --- | ---: | ---: | --- |
| internal_test | 0.9564 | 0.9526 | slight regression |
| round2_dev | 0.6220 | 0.7392 | much better hard-positive recall |
| teacher_test | 0.9133 | 0.9020 | regression |

Router conclusion:

```text
Useful as a diagnostic and as a feature, but not safe as a final decision
system. It fixed too few false negatives and induced too many human false
positives.
```

### 3.4 Phase 3: Stacking Fusion

Round2 added deployable-feature stacking:

```text
p_tfidf
p_deberta_step7
p_ensemble_step7
probability disagreement features
text length / line / punctuation / archaic / academic marker features
rule-based bucket
```

An initial leakage bug using `generator` as a feature was found and removed.
The best Step7-only stacker was:

| Split | Step7 F1 | Stacker F1 |
| --- | ---: | ---: |
| internal_test | 0.9564 | 0.9604 |
| round2_dev | 0.6220 | 0.6999 |
| teacher_test | 0.9133 | 0.9109 |

Stacker conclusion:

```text
The stacker improved internal-test and hard-dev diagnostics, but did not
generalize to the teacher-test result. The meta-training distribution was still
too different from the final test boundary.
```

### 3.5 Phase 4: RoBERTa Third Branch

RoBERTa standalone:

| Split | F1 |
| --- | ---: |
| validation | 0.9434 |
| internal_test | 0.9262 |
| round2_dev | 0.6920 |
| teacher_test | 0.8219 |

Best stacker with RoBERTa:

| Split | F1 | Confusion |
| --- | ---: | --- |
| internal_test | 0.9610 | [[851, 33], [33, 814]] |
| round2_dev | 0.7662 | [[561, 65], [126, 313]] |
| teacher_test | 0.8917 | [[126, 24], [10, 140]] |

RoBERTa conclusion:

```text
RoBERTa supplied a hard-positive recall signal, but it over-shifted the final
boundary toward predicting LLM. It reduced some false negatives but created too
many human false positives.
```

### 3.6 Round2 Final Outcome

Teacher-test final comparison:

| Candidate | Accuracy | Correct | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 274 / 300 | 0.9133 | 13 | 13 |
| bucket_routed | 0.9000 | 270 / 300 | 0.9020 | 18 | 12 |
| stacker_step7 | 0.9100 | 273 / 300 | 0.9109 | 15 | 12 |
| roberta_single | 0.8267 | 248 / 300 | 0.8219 | 22 | 30 |
| stacker_with_roberta | 0.8867 | 266 / 300 | 0.8917 | 24 | 10 |

Round2's important lesson:

```text
The task is not simply "find more LLM positives." The bottleneck is repairing
hard LLM false negatives while protecting high-style human false positives.
Round2 improved hard-dev recall, but teacher-test performance fell whenever
the method became too aggressive.
```

## 4. Round 3: Precision-Guarded Repair Attempt

Round3 was designed as a precision-guarded repair round, not a more aggressive
version of Round2.

Default policy:

```text
final_pred = step7_pred
```

New branches or rules should override Step7 only when they have strong evidence
that a Step7-human prediction is actually LLM and the text is not in a high-risk
human bucket.

### 4.1 Phase A: Error-Delta Audit

Round3 added:

```text
src/evaluation/round3_error_delta_audit.py
```

The audit showed that Round2 candidates fixed very few Step7 false negatives:

| Candidate | Fixed Step7 errors | Fixed FN | Broke Step7 correct | Induced FP |
| --- | ---: | ---: | ---: | ---: |
| bucket_routed | 1 | 1 | 5 | 5 |
| stacker_step7 | 1 | 1 | 2 | 2 |
| roberta_single | 8 | 2 | 34 | 15 |
| stacker_with_roberta | 4 | 3 | 12 | 12 |

Only these Step7 false-negative IDs were fixed by any Round2 candidate:

```text
106, 107, 249
```

Candidate-induced false positives clustered in high-risk human styles:

```text
poetry_classical
literary_short_fragment
academic_formal
polished general prose
```

This strongly supports a conservative override design.

### 4.2 Phase B: Precision-Guard Data

Round3 built precision-guard data:

```text
data/processed/round3_hard_negative_mirror_source.jsonl
data/processed/round3_llm_hardpos_multi_generator_seed.jsonl
data/processed/round3_precision_guard_train.jsonl
data/processed/round3_precision_guard_dev.jsonl
```

Checks:

| Check | Result |
| --- | ---: |
| hard human negative pool | 9762 |
| hard LLM positive pool | 2872 |
| precision-dev rows | 564 |
| precision-dev class balance | 282 human / 282 LLM |
| manual spotcheck packet | 80 rows |

Known data gap:

```text
human_literary_old_prose_mirror remained below the ideal 400-600 target.
```

### 4.3 Phase C: ELECTRA Branch

Round3 trained:

```text
model_name = google/electra-base-discriminator
train = data/processed/round3_precision_guard_train.jsonl
```

ELECTRA metrics:

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9363 | 0.9554 | 0.9125 | 0.9335 | 36 | 74 |
| internal_test | 0.9341 | 0.9587 | 0.9044 | 0.9307 | 33 | 81 |
| round3_precision_guard_dev | 0.6525 | 0.8258 | 0.3865 | 0.5266 | 23 | 173 |

ELECTRA conclusion:

```text
ELECTRA v1 did not pass the third-branch gate. It underperformed Step7 and did
not provide a safe enough override signal.
```

### 4.4 Phase D/E/F: OOF Stacker And Precision Guard

OOF stacker:

| Split | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| internal_test | 0.9440 | 0.9847 | 0.8996 | 0.9402 | 12 | 85 |
| teacher_test | 0.9033 | 0.9353 | 0.8667 | 0.8997 | 9 | 20 |

OOF conclusion:

```text
The OOF stacker became too conservative. It reduced false positives but missed
too many LLM positives.
```

Precision-guarded routing looked useful on non-teacher tuning splits:

| Split | Accuracy | Precision | Recall | F1 | FP | FN | Step7 FP | Step7 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid | 0.9659 | 0.9668 | 0.9634 | 0.9651 | 28 | 31 | 28 | 33 |
| round2_dev | 0.7822 | 0.8966 | 0.5330 | 0.6686 | 27 | 205 | 24 | 230 |
| guard_dev | 0.7624 | 0.9405 | 0.5603 | 0.7022 | 10 | 124 | 10 | 142 |

On guard-dev it fixed 18 Step7 false negatives with 0 induced false positives.
But on teacher test, the final precision guard made one override:

```text
overrides = 1
fixed Step7 FN = 0
induced FP = 1
```

Final Round3 teacher-test comparison:

| Candidate | Accuracy | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| step7 | 0.9133 | 0.9133 | 13 | 13 |
| round2_bucket_routed | 0.9000 | 0.9020 | 18 | 12 |
| round2_stacker | 0.9100 | 0.9109 | 15 | 12 |
| round2_roberta | 0.8267 | 0.8219 | 22 | 30 |
| round2_stacker_with_roberta | 0.8867 | 0.8917 | 24 | 10 |
| round3_electra | 0.8800 | 0.8759 | 13 | 23 |
| round3_oof_stacker | 0.9033 | 0.8997 | 9 | 20 |
| round3_precision_guard | 0.9100 | 0.9103 | 14 | 13 |

Round3's important lesson:

```text
Precision-guarded repair is conceptually right, but the current guard-dev set
and third-branch signals are not yet close enough to the teacher-test residual
distribution. The override rule learned a pattern that helped the constructed
guard set but did not transfer to the actual remaining teacher-test errors.
```

## 5. What The Three Rounds Prove

### 5.1 Things That Work

1. The hybrid DeBERTa + TF-IDF structure is still the best final base.
2. Targeted train-side data matters more than generic expansion.
3. Step7 neural retraining was the only optimization that truly improved the
   teacher-test result.
4. Hard-negative human examples are necessary to avoid over-predicting LLM.
5. Hard-positive LLM examples are necessary because conservative rewrites are
   the main false-negative source.
6. Error-ledger and bucket-based diagnostics are valuable for explaining
   failures and designing future data.

### 5.2 Things That Are Probably Exhausted

1. Global threshold tuning.
2. Simple alpha retuning between DeBERTa and TF-IDF.
3. Simple probability averaging among existing Step7 variants.
4. A domain router used as a standalone final classifier.
5. A direct lightweight stacker trained on a small hard-dev set.
6. RoBERTa/ELECTRA as an unguarded global replacement branch under the current
   training recipes.

The strongest evidence is the Round2 diagnostic ceiling: even oracle
teacher-test thresholds for existing prediction files only reached about
0.9267, and simple existing-family averaging only reached about 0.9333.

### 5.3 The Real Bottleneck

The remaining problem is not "overall LLM detection." It is local ambiguity:

```text
High-style human text and human-like LLM rewrites occupy the same stylistic
region. The model needs evidence that separates source from style inside that
small region.
```

The current Step7 errors are balanced:

```text
13 human false positives
13 LLM false negatives
```

That means a future candidate must satisfy a harsh budget:

```text
To reach 95%, it must reduce total errors from 26 to at most 15.
It cannot simply trade 6 fewer FN for 6 more FP.
It needs a net gain of at least +11 correct.
```

## 6. Most Plausible Route To 95%

The most plausible strict-route plan is:

```text
Data-first paired mirror expansion -> stronger calibrated neural retraining ->
local residual-error classifier -> precision-guarded override -> final strict
teacher-test diagnostic.
```

### 6.1 Build Paired Residual Buckets, Not Generic Data

For every hard LLM positive bucket, build a matched human-negative mirror
bucket with similar length, domain, era, lineation, and style.

| Hard LLM positive bucket | Matched human negative mirror |
| --- | --- |
| conservative literary rewrite | polished human literary prose |
| old-fiction style rewrite | public-domain old fiction original passages |
| archaic poetry rewrite | real archaic / classical poetry |
| free-verse rewrite | real free verse / lyrical fragments |
| natural academic paraphrase | human formal academic prose |
| short LLM fragment | human short reflective fragment |

Recommended new data target:

| Data type | Target rows | Reason |
| --- | ---: | --- |
| hard human negatives | 3000-5000 | protect against FP regressions |
| hard LLM positives | 3000-5000 | repair conservative rewrite FN |
| old-prose human mirrors | at least 800-1200 | current Round3 gap |
| poetry/freeverse mirrors | at least 1000-1500 | teacher-test errors include poetry-like human FP |
| natural academic mirrors | at least 800-1200 | academic formal text caused induced FP risk |

Key rule:

```text
new hard human negatives : new hard LLM positives >= 1 : 1
```

If this ratio is not maintained, the model will again learn "high style means
LLM" and will induce false positives.

### 6.2 Generate Hard Positives With Multiple Generators And Prompt Families

Round2 over-focused on ChatGPT-style hard positives. The next data round should
use multiple generators and prompt templates:

- ChatGPT conservative rewrite;
- DeepSeek conservative rewrite;
- Gemini conservative rewrite;
- Doubao conservative rewrite;
- old-fiction preservation;
- line-break-preserving poetry rewrite;
- natural academic paraphrase;
- minimal-edit paraphrase.

The goal is not diversity for its own sake. The goal is to prevent the model
from learning a single generator artifact.

Quality gates:

```text
No prompt leakage.
No apology or meta text.
Length ratio within a controlled range.
For poetry, preserve lineation where intended.
For prose, avoid extremely short or template-like rewrites.
Reject near duplicates that are too close to train/valid/internal-test text.
```

### 6.3 Train A Stronger Neural Branch, But Only After The Data Fix

Model priority:

1. DeBERTa-v3-base retrain with better residual buckets and sample weights.
2. DeBERTa-v3-large if hardware permits.
3. ModernBERT or ELECTRA only if it shows genuinely different error patterns.
4. A char n-gram / stylometry branch as a guard feature, not as a global
   classifier.

The strongest next candidate is probably not a fresh global ELECTRA branch. It
is more likely to be:

```text
Step7 DeBERTa architecture + much better paired residual data + sample weights
that protect hard human negatives.
```

Suggested sample weighting:

| Sample type | Weight |
| --- | ---: |
| original balanced data | 1.0 |
| hard LLM positives | 1.0 |
| high-style hard human negatives | 1.5 |
| poetry/freeverse human mirrors | 1.5-2.0 |
| old-prose human mirrors | 1.5-2.0 |
| natural academic human mirrors | 1.2-1.5 |

Promotion gate before teacher-test diagnostic:

| Gate | Required signal |
| --- | --- |
| internal-test F1 | >= 0.963 preferred |
| hard-dev recall | improves over Step7 |
| hard-negative mirror FP | not worse than Step7 |
| poetry/freeverse FP | not worse than Step7 |
| academic FP | not worse than Step7 |
| error overlap with Step7 | clearly lower than Round3 precision guard |

### 6.4 Use A Local Residual-Error Classifier Instead Of A Global Replacement

Round3 showed that a global OOF stacker became too conservative and that a
precision guard can help dev sets but miss teacher-test residuals. The next
version should be narrower:

```text
Only classify Step7-disagreement or Step7-low-confidence regions.
Do not let the new branch decide all 300 rows.
```

The residual classifier should focus on rows where:

```text
Step7 prediction = human
AND at least one new branch assigns high LLM probability
AND the text bucket is not a known high-risk human bucket
AND hard-human guard probability is low
```

For high-risk buckets, require stronger evidence:

```text
two independent branches agree
probability disagreement is low
stylometry branch supports LLM
text length is sufficient
bucket-specific hard-negative guard does not fire
```

### 6.5 Add A Human-Style Guard, Not Only An LLM Detector

The missing component is a guard that says:

```text
This looks like high-style human text, so do not override Step7 to LLM unless
evidence is overwhelming.
```

Recommended guard features:

- char n-gram TF-IDF;
- word shape and punctuation profile;
- lineation and average line length;
- archaic marker counts;
- academic marker counts;
- type-token ratio;
- source-era style markers;
- classifier trained only on hard human mirrors vs hard LLM positives.

This branch should not be optimized for overall F1. Its purpose is to veto
unsafe overrides.

### 6.6 Validate With A Better Pre-Teacher Gate

The next pre-teacher evaluation should use three holdouts:

1. original validation/internal-test;
2. residual hard-positive dev;
3. hard-human mirror dev.

A candidate should advance only if it satisfies all three:

```text
original internal-test F1 >= Step7
hard-positive FN < Step7
hard-human mirror FP <= Step7
```

This is the core correction after Round2 and Round3: hard-dev improvement alone
is not enough, and guard-dev improvement alone is not enough.

## 7. Expected Chance Of Reaching 95%

Under the strict route, 95% is difficult but not impossible. The evidence says:

| Route | Plausibility | Reason |
| --- | --- | --- |
| More threshold/alpha tuning | Very low | existing-family oracle ceiling only about 93.33 |
| Another simple stacker | Low | Round2 and Round3 stackers did not transfer |
| Another unguarded transformer branch | Low to medium | RoBERTa/ELECTRA changed errors but created instability |
| Better paired residual data + weighted DeBERTa retrain | Medium | Step7 was the only route that improved teacher-test |
| Better paired data + residual override + human-style guard | Highest | directly targets the 13 FP / 13 FN tradeoff |
| Teacher-test-aware manual tuning | High for score, low for validity | would not be a strict generalization result |

The realistic strict target for the next full round should be:

```text
first milestone: 280 / 300
second milestone: 282-284 / 300
stretch target: 285 / 300
```

The reason is simple: going from 274 to 285 requires fixing nearly half of the
current errors without breaking previously correct examples. The previous two
rounds showed that this is exactly where naive recall improvements fail.

## 8. Concrete Next Execution Plan

### Step 1: Rebuild The Residual Dataset

Create:

```text
data/processed/round4_hard_human_mirror_seed.jsonl
data/processed/round4_hard_llm_positive_seed.jsonl
data/processed/round4_residual_train.jsonl
data/processed/round4_residual_dev_hardpos.jsonl
data/processed/round4_residual_dev_hardneg.jsonl
data/processed/round4_residual_report.json
```

Acceptance:

```text
hard human negatives >= 3000
hard LLM positives >= 3000
old-prose human mirrors >= 800
poetry/freeverse human mirrors >= 1000
manual spot check >= 100 rows
no teacher-test near duplicates
```

### Step 2: Weighted DeBERTa Retrain

Use or extend:

```text
src/models/train_weighted_transformer.py
```

Train:

```text
outputs/models/round4_deberta_weighted_residual/
```

Acceptance:

```text
internal-test F1 >= 0.963 preferred
hard-positive dev recall improves
hard-human mirror dev FP does not increase
```

### Step 3: Human-Style Guard Branch

Train a lightweight guard:

```text
outputs/models/round4_human_style_guard/
```

Candidate model:

```text
char n-gram TF-IDF + logistic regression
features from src/evaluation/assign_text_bucket.py and error-ledger utilities
```

Acceptance:

```text
high-style human mirror precision is strong
guard correctly flags known induced-FP buckets
```

### Step 4: Residual Override Tuning

Default:

```text
final_pred = step7_pred
```

Allow override only for:

```text
Step7 says human
AND weighted DeBERTa says LLM strongly
AND residual classifier says LLM
AND human-style guard does not veto
```

Acceptance before teacher-test diagnostic:

```text
internal-test F1 >= Step7
hard-positive FN clearly lower than Step7
hard-human mirror FP <= Step7
overrides are explainable by bucket
```

### Step 5: Final Teacher-Test Diagnostic

Only after all non-teacher gates pass:

```text
compare step7 vs round4_candidate on teacher test
```

Promotion rule:

```text
Promote only if teacher-test correct > 274 and the change can be explained
without teacher-test label tuning.
Target promotion for 95%: >= 285 / 300.
```

## 9. Final Recommendation

Do not keep spending time on global thresholds, alpha sweeps, or larger
unguided stackers. Those routes have already shown their ceiling.

The best next attempt should be called a residual repair round:

```text
Round4 = paired residual data + weighted neural retrain + human-style guard +
precision-guarded local override.
```

If this still plateaus below 95%, the honest conclusion is that the current
available training distribution is not close enough to the teacher-test
distribution. At that point, the only remaining high-probability way to reach
95% is to add substantially more teacher-test-like but non-leaking data, or to
switch to a teacher-aware post-hoc repair route and label it clearly as such.

