# Round8 One-Shot 95% Optimization Plan

Updated: 2026-05-22

This document records the last high-risk route considered after seven rounds of
optimization. It is written as a handoff, not as a promoted result. The final
submission-grade baseline remains the Step7 DeBERTa + TF-IDF ensemble.

## 0. Executive Decision

The current strongest strict system is:

```text
DeBERTa branch: outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF branch:  outputs/models/tfidf_lit_academic_poetry
alpha:          0.5
threshold:      0.55
teacher-test:   274 / 300 correct, accuracy = 0.9133, F1 = 0.9133
```

The 95% target requires at least `285 / 300` correct predictions, so the system
would need a net gain of `+11` teacher-test-style examples over Step7.

After Round3-Round7, the following low-yield routes are exhausted:

```text
global threshold tuning
alpha tuning
extra bucket/router rules
Step7-vs-Round4 override rules
teacher-test-driven guard repair
```

The only plausible one-shot route is a distribution-level rebuild:

```text
Residual Distribution Rebuild
+ Residual-Aware DeBERTa Retraining
+ Stylometry / Surface Feature Guard
+ Ambiguous-Zone Selector
+ Strict Non-Teacher Promotion Gate
```

In short: reaching 95% would require a new residual train/dev/probe set that
resembles the remaining teacher-test errors, a residual-aware model branch, and
a local selector that only edits the ambiguous zone. More rules on top of the
existing prediction space are unlikely to generalize.

## 1. Why Step7 Is Stuck Near 91.33%

Step7 teacher-test result:

| Quantity | Value |
| --- | ---: |
| Correct | 274 / 300 |
| Errors | 26 / 300 |
| False positives | 13 |
| False negatives | 13 |

The 95% target allows at most `15 / 300` errors. The remaining errors are not a
single threshold problem; they are two opposing residual distributions.

Human false positives are concentrated in:

```text
high-style literary prose
polished literary prose
archaic or old-fashioned English
formal academic prose
modern and classical poetry
clean explanatory human text
```

These texts are fluent, clean, formal, and stylistically close to LLM output.

LLM false negatives are concentrated in:

```text
conservative ChatGPT paraphrases
old-fiction-style rewrites
academic paraphrases that preserve terminology
archaic poem rewrites
low-temperature style-preserving rewrites
light rewrites with high lexical overlap
```

These LLM examples remain too close to human-edited text. The model must fix
both sides without trading one error type for the other.

## 2. Why Seven Rounds Did Not Break Through

### Step7 Is Already A Local Optimum

The original final ensemble had:

```text
accuracy = 0.9033
FP = 21
FN = 8
```

Step7 moved to:

```text
accuracy = 0.9133
FP = 13
FN = 13
```

It reduced human false positives substantially, but at the cost of more LLM
false negatives. A simple threshold move now mostly swaps FP for FN.

### Later Rounds Stayed Inside A Correlated Prediction Space

Round3-Round7 mainly reused combinations of these signals:

```text
Step7 ensemble
Round4 residual DeBERTa branch
Round5 unsafe guard
Round7 exact selector
OOF stacker
precision guard
hard-positive / hard-negative residual pools
```

These signals are useful diagnostically, but they do not add enough independent
information. The common failure mode was:

```text
local hard-dev improves
teacher-test does not improve
```

### Safe Teacher-Test Override Surface Is Too Small

Round7's exact override surface exposed the core bottleneck:

```text
safe fixed-FN candidates = 3
unsafe induced-FP candidates = 14
```

Relaxing the unsafe guard might fix a few LLM false negatives, but it would
likely introduce human false positives. The net gain is unlikely to reach +11.

## 3. Non-Negotiable Guardrails

Teacher-test is diagnostic only. The following are prohibited:

```text
using teacher_test text in training
adding teacher-test errors to residual train/dev
generating rewrites from teacher-test text for training
using near duplicates of teacher-test examples in train/dev
training selectors, guards, routers, or thresholds from teacher-test labels
```

Teacher-test results must not be used to tune:

```text
thresholds
alpha
selector thresholds
unsafe-guard thresholds
bucket-specific rules
override conditions
promotion gates
```

Before any teacher-test diagnostic, freeze:

```text
training data
model checkpoints
thresholds
routing rules
selector/guard parameters
promotion criteria
```

The promotion gate must be built from validation, internal-test, hard-positive,
hard-negative, and residual probe data. Teacher-test is a final readout only.

## 4. One-Shot Strategy

The next viable route would be:

1. Define a residual error taxonomy.
2. Build teacher-test-like residual candidate pools without using teacher-test
   text or labels for training.
3. Run Step7 on the candidate pool to select hard examples.
4. Generate conservative LLM hard positives.
5. Construct high-style human hard negatives.
6. Train a residual-aware DeBERTa branch, preferably continuing from Step7.
7. Train a stylometry / surface-feature branch.
8. Train an ambiguous-zone selector.
9. Apply a strict non-teacher gate.
10. Run teacher-test once only if the non-teacher gate strongly beats Step7.

The key design principle is not to revise every sample. The default prediction
should remain Step7, and the repair system should only operate in an ambiguous
zone, for example:

```text
0.35 <= P_step7 <= 0.70
```

or the narrower local surface:

```text
Step7 predicts human
residual branch predicts LLM
selector predicts safe override
```

High-confidence Step7 predictions are mostly correct and should not be touched.

## 5. Residual Data Requirements

The residual pool should include high-style human hard negatives and
conservative LLM hard positives across:

| Bucket | Purpose |
| --- | --- |
| high-style literary prose | protect polished human passages |
| old prose / archaic English | cover old-fiction residuals |
| classical poetry | cover poetic structure and archaic diction |
| free verse | cover line-break and modern poetic form |
| formal academic prose | protect clean human academic writing |
| conservative ChatGPT rewrite | target the hardest LLM positives |
| old-fiction LLM rewrite | target style-preserving literary rewrites |
| academic paraphrase | target terminology-preserving rewrites |
| short fragment | target under-contextualized examples |

Every row should preserve:

```text
source_id / pair_id
label
bucket
generation method, when applicable
Step7 probability and prediction
residual branch probability and prediction
split assignment
leakage checks
```

The split must be group-safe by source or pair. The probe split must be used
only for the final non-teacher gate, not for tuning.

## 6. Residual-Aware DeBERTa

Recommended starting point:

```text
base checkpoint: outputs/models/deberta_lit_academic_poetry_step7_combined/best_model
train mix:       original train + controlled residual hard examples
learning rate:   8e-6 to 1e-5
epochs:          2 to 3
```

The residual branch must be evaluated on:

```text
original validation
original internal_test
residual_dev
residual_probe
hard-positive dev
hard-negative dev
```

If residual performance improves while original internal-test human precision
falls materially, reject the branch. The goal is local repair, not a global
replacement.

## 7. Stylometry / Surface Branch

The stylometry branch is the most important new independent signal. It should
not be used as the main classifier. It should provide features for the
ambiguous-zone selector and help separate:

```text
high-style human passages misread by transformers
conservative LLM rewrites with subtle surface artifacts
```

Suggested features:

```text
word and character n-grams
sentence length statistics
line-break density
punctuation profile
archaic marker counts
function-word ratios
type-token ratio
readability statistics
quote/dialogue markers
academic marker counts
```

Start with a stable, interpretable model such as logistic regression or a small
linear classifier. The branch only needs to add orthogonal evidence, not win as
a standalone detector.

## 8. Promotion Gate

A candidate may advance to a teacher-test diagnostic only if it passes strict
non-teacher gates:

| Gate | Requirement |
| --- | --- |
| Original internal-test F1 | no meaningful regression versus Step7 |
| Original internal-test FP | no material FP increase |
| Hard-negative FP | no increase versus Step7 |
| Residual-dev fixed FN | strong positive gain |
| Residual-probe fixed FN | positive gain without overfitting |
| Residual-probe induced FP | zero or tightly bounded |
| Leakage audit | no teacher-test text or near-duplicate training |

The candidate should also have a clear frozen prediction contract:

```text
default = Step7
override only within ambiguous zone
override only when selector predicts safe
never tune thresholds from teacher-test
```

## 9. Final Recommendation

Do not continue with another rule-only Round8. The final baseline should remain
Step7 unless a residual-aware branch plus stylometry selector clears the
non-teacher gate by a large margin.

For final project reporting, the most defensible conclusion is:

```text
Step7 is the validated final baseline.
Later rounds produced useful diagnostic and training assets, but they did not
produce a safer deployable model.
The ceiling is limited by residual distribution mismatch and highly correlated
model errors, not by a missing threshold tweak.
```
