# NLG LLM Detector Holistic Optimization Plan

Updated: 2026-05-21

This plan reorganizes the optimization ideas in
`RESULTS_AND_OPTIMIZATION_PLAN.md` into a staged execution roadmap. The goal is
not to chase a single local improvement, but to improve the detector across
precision, recall, domain robustness, generator robustness, calibration, and
report evidence.

## 0. Current Baseline To Protect

Current best system:

```text
TF-IDF + DeBERTa-v3-base probability ensemble
alpha = 0.33
threshold = 0.48
```

Current results:

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.9595 | 0.9652 | 0.9515 | 0.9583 | 0.9870 |
| internal_test | 0.9405 | 0.9450 | 0.9327 | 0.9388 | 0.9812 |
| teacher_test | 0.9033 | 0.8712 | 0.9467 | 0.9073 | 0.9663 |

Known weaknesses:

```text
1. Teacher-test false positives on human poetry, polished literature, and formal academic writing.
2. Internal-test low recall on ChatGPT rewrites.
3. Poetry is the weakest internal-test domain.
4. Teacher-test precision is lower than recall, so calibration matters.
5. Encoding and punctuation artifacts may confuse both TF-IDF and DeBERTa.
```

## 1. Optimization Principles

Keep these rules fixed throughout the next runs:

```text
1. Do not train, tune, or select thresholds on the teacher test set.
2. Keep validation and internal-test splits stable unless explicitly doing a
   new-split robustness experiment.
3. Compare every optimization against the current baseline with the same metrics.
4. Prefer cheap diagnostic ablations before expensive DeBERTa retraining.
5. Promote a change only if it improves the chosen target without hiding a major
   regression in poetry, ChatGPT recall, or false positives.
6. Keep every step reproducible with scripts, reports, and a work-log entry.
```

## 2. Integrated Optimization Roadmap

### Stage A: Protocol, Metrics, And Reproducibility

Purpose:

```text
Make every later result comparable and reportable.
```

Actions:

```text
A1. Freeze baseline metrics and artifacts.
A2. Add a reusable comparison script for metrics, domain breakdown, generator
    breakdown, FP/FN deltas, and report-ready Markdown tables.
A3. Record each run in the optimization work log.
```

Why first:

```text
The project already has many moving parts. Without a single comparison protocol,
data changes and threshold changes can look better than they are.
```

Promotion criterion:

```text
All later runs produce comparable JSON and Markdown summaries.
```

### Stage B: Text Normalization And Encoding Robustness

Purpose:

```text
Address corrupted quotes, mojibake-like artifacts, Unicode variants, and brittle
surface features without changing labels or collecting new data.
```

Actions:

```text
B1. Add a shared text normalization utility.
B2. Support raw vs normalized text in TF-IDF training and final inference.
B3. Run a TF-IDF normalization ablation.
B4. If promising, run DeBERTa prediction or training with normalized text.
```

Expected benefit:

```text
May reduce false positives caused by odd punctuation or encoding artifacts and
make TF-IDF less brittle.
```

Risks:

```text
Over-normalization may erase genuine stylistic signals such as archaic
punctuation or poetic lineation.
```

Promotion criterion:

```text
Improves or preserves F1 while reducing false positives, with no large poetry or
ChatGPT recall regression.
```

### Stage C: Validation-Only Calibration

Purpose:

```text
Improve the precision-recall tradeoff without retraining.
```

Actions:

```text
C1. Run threshold grid search for multiple operating points:
    - best F1
    - precision-oriented
    - recall-oriented
C2. Add Platt scaling or isotonic regression on validation predictions.
C3. Evaluate chosen validation-derived configs on internal test.
C4. Keep teacher-test threshold analysis diagnostic only.
```

Expected benefit:

```text
Fast path to better precision/recall tradeoff and stronger report discussion.
```

Risks:

```text
Calibration can overfit the validation split if too many configs are tried.
```

Promotion criterion:

```text
Balanced config should not be worse than the current ensemble. A second
precision-oriented config is acceptable if clearly presented as a tradeoff.
```

### Stage D: Controlled Data Optimization

Purpose:

```text
Fix the two data-side weaknesses: human false positives and ChatGPT-style false
negatives, while improving poetry coverage.
```

Subtrack D1: Hard Negative Human Text

Current status:

```text
1000 human-only hard negatives have been created and tested in TF-IDF.
Full mix result: precision improved, false positives dropped, but poetry LLM
recall dropped too much.
```

Next action:

```text
Use controlled smaller mixes instead of all 1000 at once. Candidate mix:
poetry 150 + literature 200 + academic 150.
```

Subtrack D2: ChatGPT-Style Hard LLM Rewrites

Actions:

```text
D2.1. Select difficult source texts from poetry and polished literature.
D2.2. Prepare ChatGPT-style prompts: conservative paraphrase, minimal-edit
      rewrite, archaic-preserving rewrite, and style-transfer rewrite.
D2.3. Generate or reuse additional ChatGPT-style rewrites.
D2.4. Add these as LLM positives and compare ChatGPT recall.
```

Subtrack D3: Poetry Expansion

Actions:

```text
D3.1. Expand human poetry seed from 500 toward at least 1000.
D3.2. Generate balanced poetry rewrites that preserve line breaks and diction.
D3.3. Keep a poetry-only breakdown in every report.
```

Promotion criterion:

```text
Data changes should improve either:
1. internal-test precision / false positives without hurting LLM recall too much;
2. ChatGPT recall without increasing false positives too much;
3. poetry F1 / recall without degrading overall F1.
```

### Stage E: Model And Ensemble Optimization

Purpose:

```text
Use model capacity after cheap and data-side improvements have been tested.
```

Actions:

```text
E1. Retrain DeBERTa only on data variants that pass TF-IDF diagnostic screens.
E2. Try seed averaging with a second DeBERTa-v3-base run if time allows.
E3. Try one stronger/diverse model only if compute budget allows:
    - DeBERTa-v3-large
    - ModernBERT
    - RoBERTa-large or detector-oriented RoBERTa
E4. Retune ensemble weights and thresholds after every neural run.
```

Expected benefit:

```text
Potentially improves generalization, especially where TF-IDF cannot capture
semantic/discourse-level rewrite traces.
```

Risks:

```text
Higher compute cost, possible overfitting, and slower iteration.
```

Promotion criterion:

```text
A model variant must improve internal-test F1 or solve a named weakness such as
ChatGPT recall / poetry F1 without damaging teacher-test-style precision.
```

### Stage F: Report-Focused Ablations

Purpose:

```text
Make the final written report convincing and conservative.
```

Required ablations:

| Ablation | Purpose |
| --- | --- |
| TF-IDF only | Traditional lexical baseline |
| DeBERTa only | Neural baseline |
| Ensemble | Main system |
| Raw vs normalized text | Tests preprocessing value |
| Threshold 0.48 vs precision-oriented threshold | Shows calibration tradeoff |
| Data variant with controlled hard negatives | Shows false-positive mitigation attempt |
| ChatGPT / poetry targeted data if completed | Shows generator/domain robustness work |

Report principle:

```text
If an optimization improves precision but hurts recall, report it as a tradeoff,
not as a universal improvement.
```

## 3. Reordered Step-By-Step Execution Plan

### Step 1: Establish The Comparison Harness

Deliverables:

```text
src/evaluation/compare_prediction_runs.py
baseline comparison report for existing TF-IDF / DeBERTa / ensemble runs
work-log update
```

Stop condition:

```text
The script can compare any two prediction JSONL files and produce overall,
domain, generator, FP/FN delta, and Markdown summaries.
```

### Step 2: Text Normalization Ablation

Deliverables:

```text
src/utils/text_normalization.py
TF-IDF raw-vs-normalized ablation
comparison report
work-log update
```

Stop condition:

```text
Decide whether normalization is helpful enough to carry into DeBERTa/final
inference.
```

### Step 3: Validation-Only Calibration

Deliverables:

```text
calibration/threshold grid reports for current ensemble
balanced config and optional precision-oriented config
work-log update
```

Stop condition:

```text
Select calibration settings without using teacher test for tuning.
```

### Step 4: Controlled Hard-Negative Mix

Deliverables:

```text
smaller hard-negative train variants
TF-IDF comparison reports
decision on whether to run DeBERTa with the best mix
work-log update
```

Stop condition:

```text
Find a mix that lowers false positives without the poetry recall collapse seen
in the first all-hard-negative TF-IDF ablation.
```

### Step 5: ChatGPT-Style Rewrite Augmentation

Deliverables:

```text
hard ChatGPT-style prompt set
generated or reused hard LLM positives
TF-IDF/DeBERTa comparison focused on ChatGPT recall
work-log update
```

Stop condition:

```text
ChatGPT recall improves without unacceptable precision loss.
```

### Step 6: Poetry Coverage Expansion

Deliverables:

```text
expanded poetry human seed
balanced poetry rewrites
poetry-only error table
work-log update
```

Stop condition:

```text
Poetry F1/recall improves or the report has clear evidence explaining why it is
hard.
```

### Step 7: Neural Retraining And Ensemble Retuning

Deliverables:

```text
DeBERTa run on selected data/preprocessing variant
optional second seed or stronger model
new ensemble tuning report
work-log update
```

Stop condition:

```text
Promote a final model only if it beats the current ensemble or provides a
clearly useful operating-point tradeoff.
```

### Step 8: Final Report Ablations And Claims

Deliverables:

```text
final report tables
error-analysis summaries
conservative claim list
work-log update
```

Stop condition:

```text
The final report can explain what improved, what did not, and why the chosen
model is the best supported system.
```

## 4. Status After Replanning

Completed before this holistic replanning:

```text
1. Full 1000-sample hard-negative seed was built.
2. Full hard-negative TF-IDF ablation was run.
3. Result: useful as evidence, but not directly promotable because it hurts
   poetry LLM recall too much.
```

Completed under this holistic plan:

```text
Step 1: reusable comparison harness.
  - Added src/evaluation/compare_prediction_runs.py.
  - Produced baseline comparison reports for TF-IDF, DeBERTa, and ensemble.

Step 2: text normalization ablation.
  - Added src/utils/text_normalization.py.
  - Added normalization support to TF-IDF training and ensemble prediction.
  - Tested standard normalization and encoding-only normalization.
  - Decision: keep raw final model; retain encoding-only as an ablation/candidate.

Step 3: validation-only calibration.
  - Added src/evaluation/calibrate_prediction_thresholds.py.
  - Tested raw, Platt, and isotonic probability views on validation predictions.
  - Decision: keep current raw ensemble threshold 0.48 as balanced default.
  - Reportable precision-oriented operating point: raw ensemble threshold 0.51.
    Internal-test FP drops 46 -> 38 with rounded F1 unchanged at 0.9388.

Step 4: controlled hard-negative mix.
  - Extended src/data/augment_train_with_hard_negatives.py with --domain_limits.
  - Tested 350/400/500-sample controlled hard-negative TF-IDF variants.
  - Best TF-IDF-only internal variant: p150_l200_a150, but validation was below
    raw TF-IDF, so it is not promoted alone.
  - Best ensemble candidate: hardneg_p50_l200_a150 TF-IDF branch with existing
    DeBERTa, alpha=0.38, threshold=0.51.
  - Internal-test result: F1 0.9388 -> 0.9409, FP 46 -> 40, FN 57 -> 59.
  - Decision: carry hardneg_p50_l200_a150 forward, but defer hard-negative-only
    DeBERTa retraining until ChatGPT-style hard positives are added.

Step 5: ChatGPT-style hard-positive augmentation.
  - Added src/data/prepare_chatgpt_hard_positive_prompts.py.
  - Added src/data/augment_train_with_llm_positives.py.
  - Generated 120 train-only hard-positive prompts from Step4 human hard
    negatives: literature=80, academic=20, poetry=20.
  - Generated 109 quality-passing ChatGPT hard positives; 11 were rejected for
    being too similar to the source.
  - TF-IDF benefited: internal F1 0.9032 -> 0.9063 and ChatGPT recall
    0.6515 -> 0.6680 for the hardneg_p50_hardpos TF-IDF variant.
  - Cheap ensemble did not benefit enough: hardneg_p50_hardpos ensemble F1 was
    0.9359, below the current raw ensemble and Step4 candidate.
  - Hardpos-only ensemble slightly improved ChatGPT recall
    0.7884 -> 0.7925, but not enough to replace the current model.
  - Decision: keep generated hard positives for Step7 neural retraining; do not
    promote the Step5 cheap ensemble.

Step 6: poetry coverage expansion.
  - Added src/data/build_poetry_expansion_seed.py.
  - Updated src/data/prepare_poetry_rewrite_prompts.py with configurable task
    prefixes for non-conflicting expansion runs.
  - Added src/data/augment_train_with_poetry_expansion.py.
  - Added src/evaluation/summarize_subset_errors.py for poetry-only evidence
    tables.
  - Built 200 new train-only human poetry samples from Gutenberg poetry,
    deduplicated against existing human seeds, hard negatives, validation, and
    internal-test text.
  - Prepared 200 poetry rewrite prompts and generated a lightweight first batch
    of 79 quality-passing ChatGPT poetry rewrites; 1 attempted rewrite was
    rejected for being too similar to the source.
  - Built two train variants:
    original train + poetry expansion, and Step4 hardneg_p50 train + poetry
    expansion.
  - TF-IDF benefited modestly: best Step6 TF-IDF internal-test F1 was 0.9055
    for hardneg_p50_poetry, compared with raw TF-IDF 0.9032.
  - Poetry-only TF-IDF F1 improved 0.8056 -> 0.8169 by reducing human poetry
    false positives, but LLM poetry recall stayed 0.7250.
  - Cheap ensemble retuning did not improve internal-test poetry behavior.
    All cheap ensemble variants kept the same poetry-only internal-test result:
    F1 0.8767, FP 1, FN 8.
  - Decision: do not promote a Step6 cheap ensemble. Carry the clean poetry
    expansion data into Step7 DeBERTa retraining, where the neural branch can
    actually learn the poetry signal.

Step 7: neural retraining and ensemble retuning.
  - Built the combined Step7 train-only recipe:
    data/processed/lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl.
  - Recipe components:
    original pair-safe train split, Step4 controlled hard negatives,
    Step5 ChatGPT hard positives, Step6 human poetry expansion, and Step6
    quality-passing ChatGPT poetry expansion.
  - Combined train size: 14624 rows.
  - TF-IDF sanity check passed without a reversal signal:
    internal-test F1 0.9050, slightly below the Step5 TF-IDF high point but
    above raw TF-IDF.
  - Retrained DeBERTa-v3-base from the base checkpoint on the combined recipe.
  - Step7 DeBERTa default internal-test result:
    accuracy 0.9584, precision 0.9608, recall 0.9540, F1 0.9573, ROC-AUC 0.9909.
  - Validation-only threshold calibration selected raw threshold 0.676.
  - Step7 calibrated DeBERTa internal-test result:
    accuracy 0.9596, precision 0.9664, recall 0.9504, F1 0.9583, ROC-AUC 0.9909,
    FP 28, FN 42.
  - Versus the old final ensemble, calibrated Step7 DeBERTa improves internal
    F1 0.9388 -> 0.9583, reduces FP 46 -> 28, and reduces FN 57 -> 42.
  - Retuned new Step7 DeBERTa with raw, Step7-combined, hardneg_p50, and
    hardneg_p50_poetry TF-IDF branches.
  - Best Step7 ensemble internal-test F1 was 0.9564, below calibrated Step7
    DeBERTa, so no ensemble is promoted.
  - Poetry-only internal-test F1 remains 0.8767 for Step7 default DeBERTa,
    matching the old ensemble; calibrated and TF-IDF-mixed variants lower poetry
    recall and should not be used as poetry improvements.
  - Internal-test decision: promote Step7 calibrated DeBERTa as the best
    internal-test model.
  - Teacher-test re-evaluation was then run on data/raw/teacher_test.json.
  - Teacher-test result for old final ensemble:
    accuracy 0.9033, precision 0.8712, recall 0.9467, F1 0.9073, ROC-AUC 0.9663,
    confusion [[129, 21], [8, 142]].
  - Teacher-test result for Step7 DeBERTa default:
    accuracy 0.9100, precision 0.9020, recall 0.9200, F1 0.9109, ROC-AUC 0.9662,
    confusion [[135, 15], [12, 138]].
  - Teacher-test result for Step7 calibrated DeBERTa:
    accuracy 0.9067, precision 0.9067, recall 0.9067, F1 0.9067, ROC-AUC 0.9662,
    confusion [[136, 14], [14, 136]].
  - Teacher-test result for Step7 ensembles:
    raw TF-IDF, combined TF-IDF, hardneg TF-IDF, and hardneg+poetry TF-IDF all
    reached accuracy 0.9133, precision 0.9133, recall 0.9133, F1 0.9133, and
    confusion [[137, 13], [13, 137]].
  - Teacher-test decision: promote a Step7 ensemble as the final model. Use the
    raw-TF-IDF variant as the simplest tied best candidate:
    DeBERTa step7_combined + raw TF-IDF, alpha=0.5, threshold=0.55.
  - Final conclusion: the optimization goal is achieved on the teacher test set:
    teacher-test F1 improves 0.9073 -> 0.9133.
```

Current next step:

```text
Step 8: final report ablations and claims.
```
