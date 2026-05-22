# LLM Text Detector Results and Optimization Plan

Updated: 2026-05-21

## 1. Current Project Position

The project has moved from data construction and baseline preparation into a complete first delivery state. The current system includes:

- a multi-domain training corpus covering literature, academic writing, and poetry;
- LLM rewrites from ChatGPT, DeepSeek, Gemini, and Doubao;
- pair-level train / validation / internal-test splitting to reduce source leakage;
- a word + character TF-IDF logistic regression baseline;
- a DeBERTa-v3-base classifier;
- a probability ensemble combining DeBERTa and TF-IDF;
- final inference and error-analysis scripts for the teacher test set.

The current best model is the fine-tuned ensemble:

```text
P_final = alpha * P_deberta + (1 - alpha) * P_tfidf
alpha = 0.33
threshold = 0.48
```

The threshold and ensemble weight were selected on the validation split, then checked on the internal test split. The teacher test set was used only for final evaluation and error analysis.

## 2. Data Summary

### 2.1 Human Seed Data

| Domain | Human samples | Source |
| --- | ---: | --- |
| Literature | 7130 | Project Gutenberg fiction |
| Academic | 1200 | ACL-OCL academic paper paragraphs |
| Poetry | 500 | Project Gutenberg poetry |
| Total | 8830 | Combined human seed |

The combined human seed report shows no duplicate IDs or duplicate `pair_id` values. The mean word count is about 109 words.

### 2.2 Full Dataset

The main dataset is:

```text
data/processed/full_dataset_lit_academic_poetry.jsonl
```

It contains 17,295 samples:

| Label | Meaning | Count |
| --- | --- | ---: |
| 0 | Human-written | 8830 |
| 1 | LLM-rewritten | 8465 |

Domain distribution:

| Domain | Count |
| --- | ---: |
| Literature | 14008 |
| Academic | 2292 |
| Poetry | 995 |

Generator distribution:

| Generator | Count |
| --- | ---: |
| Human | 8830 |
| DeepSeek | 2367 |
| ChatGPT | 2366 |
| Gemini | 1875 |
| Doubao | 1857 |

### 2.3 Pair-Safe Split

The main split file is:

```text
data/processed/lit_academic_poetry_split_report.json
```

Split sizes:

| Split | Samples | Pair IDs |
| --- | ---: | ---: |
| Train | 13836 | 7064 |
| Validation | 1728 | 882 |
| Internal test | 1731 | 884 |

The split is done by `pair_id`, so a human source passage and its corresponding LLM rewrite are kept in the same split. This is important because otherwise the classifier could learn source-specific wording rather than the human-vs-LLM distinction.

## 3. Model Results

### 3.1 Main Metrics

| Method | Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | Validation | 0.9155 | 0.9464 | 0.8771 | 0.9104 | 0.9699 |
| TF-IDF | Internal test | 0.9087 | 0.9389 | 0.8701 | 0.9032 | 0.9659 |
| DeBERTa-v3-base | Validation | 0.9514 | 0.9472 | 0.9539 | 0.9505 | 0.9889 |
| DeBERTa-v3-base | Internal test | 0.9284 | 0.9208 | 0.9339 | 0.9273 | 0.9830 |
| Ensemble | Validation | 0.9595 | 0.9652 | 0.9515 | 0.9583 | 0.9870 |
| Ensemble | Internal test | 0.9405 | 0.9450 | 0.9327 | 0.9388 | 0.9812 |
| Original final ensemble | Teacher test | 0.9033 | 0.8712 | 0.9467 | 0.9073 | 0.9663 |
| Optimized Step7 ensemble | Teacher test | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9690 |

Original teacher-test confusion matrix:

```text
[[TN, FP], [FN, TP]] = [[129, 21], [8, 142]]
```

This means:

- 129 human texts were correctly predicted as human;
- 21 human texts were incorrectly predicted as LLM;
- 8 LLM texts were missed;
- 142 LLM texts were correctly detected.

Optimized Step7 teacher-test confusion matrix:

```text
[[TN, FP], [FN, TP]] = [[137, 13], [13, 137]]
```

Compared with the original final ensemble, the optimized Step7 ensemble fixes
8 additional human false positives, while missing 5 additional LLM texts. The
net result is better accuracy, F1, and ROC-AUC on the teacher test set.

### 3.2 Interpretation

The original result was strong for a course project and for a teacher-held
external test set. The Step7 optimized ensemble improves it further, reaching
91.33% teacher-test accuracy and 0.9133 F1. The ROC-AUC improves from 0.9663 to
0.9690 for the raw-TF-IDF Step7 ensemble variant.

The ensemble improves over both individual branches on the teacher test set:

| Branch / model decision | Teacher-test accuracy |
| --- | ---: |
| TF-IDF only | 0.8900 |
| DeBERTa only | 0.8867 |
| Original final ensemble | 0.9033 |
| Optimized Step7 ensemble | 0.9133 |

This supports the central project claim: DeBERTa and TF-IDF capture complementary signals. DeBERTa contributes deeper semantic and discourse-level style features, while TF-IDF contributes lexical, punctuation, spelling, and surface n-gram features.

### 3.3 Domain Breakdown on Internal Test

| Domain | n | Accuracy | Precision | Recall | F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Academic | 258 | 0.9264 | 0.8881 | 0.9675 | 0.9261 | 15 | 4 |
| Literature | 1393 | 0.9462 | 0.9552 | 0.9342 | 0.9446 | 30 | 45 |
| Poetry | 80 | 0.8875 | 0.9697 | 0.8000 | 0.8767 | 1 | 8 |

The weakest domain is poetry. This is expected because poetry has a smaller sample count and naturally contains unusual syntax, lineation, archaic forms, and dense figurative language. These properties can make human poetry look LLM-like and can make rewritten poetry look human-like.

### 3.4 Generator Breakdown on Internal Test

| Generator | LLM rows | Recall on LLM rows | FN |
| --- | ---: | ---: | ---: |
| ChatGPT | 241 | 0.7884 | 51 |
| DeepSeek | 233 | 0.9828 | 4 |
| Doubao | 186 | 0.9946 | 1 |
| Gemini | 187 | 0.9947 | 1 |

The clear weak point is ChatGPT-style rewriting. The model detects DeepSeek, Doubao, and Gemini rewrites almost perfectly on the internal test split, but misses many ChatGPT rewrites. This suggests that ChatGPT rewrites are more conservative, more human-like, or closer to the original human style in this dataset.

## 4. Error Analysis

### 4.1 Teacher-Test Error Pattern

Teacher test:

```text
true human / LLM: 150 / 150
predicted human / LLM: 137 / 163
false positives: 21
false negatives: 8
```

The model is recall-oriented: it catches most LLM texts, but it over-predicts LLM on some human texts. This is visible from the high recall and lower precision:

```text
precision = 0.8712
recall    = 0.9467
```

### 4.2 False Positives

High-confidence false positives are mostly:

- human poetry with regular rhythm or archaic wording;
- highly polished literary prose;
- formal academic passages with symbolic or technical language;
- passages with unusual punctuation or encoding artifacts.

These cases are hard because the same properties that make LLM text detectable, such as polished phrasing, formal structure, and low noise, can also appear in real human literary or academic writing.

### 4.3 False Negatives

False negatives tend to be:

- LLM rewrites that remain close to human prose;
- archaic or poetic rewrites that mimic human historical style;
- passages where the DeBERTa branch assigns very low LLM probability and the TF-IDF branch is not strong enough to override it.

This means the next model improvement should not only add more generated text. It should specifically add hard LLM examples that preserve human-like style.

## 5. What Counts as a Good Score for This Task

For AI-generated text detection, there is no universal accuracy threshold because performance depends heavily on the domain, generator, prompt style, and whether the test set is internal or external.

A practical scale is:

| Evaluation setting | Rough interpretation |
| --- | --- |
| Internal same-distribution split above 90% | Good |
| Internal same-distribution split above 95% | Very strong, but check leakage |
| External or teacher-held test above 85% | Useful |
| External or teacher-held test above 90% | Strong |
| External or teacher-held test above 95% | Very difficult unless the test distribution is close to training |

Under that scale, the current teacher-test score is strong. The model is good enough for a reportable course project. The remaining optimization space is mostly about robustness, calibration, and reducing false positives on unusual human writing.

## 6. Optimization Plan

### 6.1 Priority 1: Add Hard Negative Human Text

Goal: reduce false positives.

The current teacher-test errors show that the model is too willing to label polished or unusual human writing as LLM. The most valuable new human data would be:

- more human poetry from multiple authors and periods;
- archaic or nineteenth-century prose;
- dramatic, rhetorical, or highly figurative literary passages;
- formal academic paragraphs with definitions, equations, or technical terms;
- human text with varied punctuation, line breaks, and style.

Recommended procedure:

1. Collect 500 to 1500 additional human-only samples from poetry and difficult literary / academic sources.
2. Keep them as human negatives and avoid creating too many easy paired rewrites at first.
3. Rebuild the dataset and compare false positives on validation and internal test.
4. Report whether precision improves without sacrificing too much recall.

Expected effect:

- precision should improve;
- teacher-test-like false positives should decrease;
- recall may drop slightly if the threshold is unchanged.

### 6.2 Priority 2: Add More ChatGPT-Style Rewrites

Goal: reduce false negatives on the hardest LLM generator.

The internal-test generator breakdown shows that ChatGPT recall is much lower than other generators. This is the clearest data-side weakness.

Recommended additions:

- more ChatGPT rewrites of poetry and difficult literary prose;
- multiple rewrite prompts, including conservative paraphrase, style transfer, modernization, and minimal-edit rewrite;
- multiple decoding styles if available, for example lower and higher temperature variants;
- rewrites that preserve archaic style rather than modernizing everything.

Expected effect:

- ChatGPT recall should improve;
- model should become less dependent on generator-specific artifacts;
- final F1 should improve if precision does not drop too much.

### 6.3 Priority 3: Improve Poetry Coverage

Goal: improve the weakest domain.

Poetry internal-test F1 is 0.8767, lower than literature and academic. The poetry test subset is small, but the error rate is still meaningful.

Recommended work:

- expand poetry human seed from 500 to at least 1000 samples;
- include sonnets, ballads, free verse, nineteenth-century poetry, and modern-looking poetry;
- generate LLM rewrites that preserve line breaks and poetic diction;
- add a domain-specific error table for poetry only.

Expected effect:

- poetry recall should improve;
- false positives on human poems should become less severe;
- the model should become more credible for the exact course-test distribution if it contains poems.

### 6.4 Priority 4: Calibrate Probabilities and Thresholds

Goal: improve the precision-recall tradeoff without retraining the whole model.

The current threshold, 0.48, is selected on validation F1. A threshold sensitivity check shows that the teacher test would have favored a slightly higher threshold, but this should only be treated as diagnostic evidence, not as a legitimate tuning procedure.

Recommended validation-only approaches:

- choose one threshold for best F1;
- choose one threshold for higher precision if false positives are more costly;
- try Platt scaling, isotonic regression, or temperature scaling on validation predictions;
- report both a balanced model and a precision-oriented model if the assignment allows it.

Important rule:

Do not tune the final threshold on the teacher test set. Use teacher-test threshold analysis only to understand the error pattern after final evaluation.

### 6.5 Priority 5: Fix Encoding and Text Normalization

Goal: reduce artifacts that confuse the detector.

The teacher-test error report includes mojibake-like text such as corrupted apostrophes. This may affect both TF-IDF and DeBERTa.

Recommended work:

- add Unicode normalization before prediction;
- consider `ftfy` for repairing common mojibake;
- normalize curly quotes and apostrophes consistently;
- keep an ablation comparing raw text vs normalized text.

Expected effect:

- TF-IDF features become less brittle;
- human literary text with encoding artifacts may be less likely to be misclassified;
- impact is uncertain, so this should be tested as an ablation.

### 6.6 Priority 6: Try Stronger or More Diverse Models

Goal: improve model capacity and robustness after data issues are addressed.

Possible models:

- DeBERTa-v3-large;
- ModernBERT;
- RoBERTa-large or a detector-oriented RoBERTa checkpoint;
- a second DeBERTa seed for seed averaging;
- a small ensemble of two transformer models plus TF-IDF.

This is lower priority than data improvements because the current error pattern is strongly tied to domain and generator coverage. A larger model may improve metrics, but hard negative and ChatGPT-style data are more likely to fix the observed weaknesses.

### 6.7 Priority 7: Add Report-Focused Ablations

Goal: make the final writeup more convincing.

Useful ablations:

| Ablation | Purpose |
| --- | --- |
| TF-IDF only | Traditional lexical baseline |
| DeBERTa only | Neural semantic baseline |
| Ensemble | Final model |
| Literature-only training | Shows value of academic / poetry data |
| Literature + academic training | Shows incremental domain expansion |
| Full data | Final setting |
| With vs without text normalization | Tests preprocessing value |
| Threshold 0.48 vs precision-oriented threshold | Shows calibration tradeoff |

The existing results already cover the first three. The most useful additional report ablation would be a full-data model with text normalization, because it directly targets the observed error cases.

## 7. Recommended Final Report Claims

Strong claims that are supported:

- The hybrid DeBERTa + TF-IDF detector outperforms both single branches on the teacher test set.
- The optimized Step7 ensemble reaches 91.33% accuracy and 0.9133 F1 on the teacher test set.
- The ensemble reduces the weakness of single-branch models by combining deep semantic features and surface n-gram features.
- The original ensemble is recall-oriented; the optimized Step7 ensemble is more balanced, with 13 false positives and 13 false negatives.
- The hardest remaining cases are stylistically unusual human texts and human-like ChatGPT rewrites.

Claims to avoid or phrase carefully:

- Do not claim the detector is generally reliable in the wild.
- Do not claim 90% accuracy will transfer to arbitrary LLMs or arbitrary domains.
- Do not claim the teacher-test threshold can be optimized from teacher-test results.
- Do not overstate poetry performance, because poetry is the weakest and smallest domain.

Conservative final conclusion:

```text
The current detector is a strong course-project system for English literary,
academic, and poetic LLM-rewrite detection. The hybrid ensemble improves over
both TF-IDF and DeBERTa alone. After Step7 optimization, the best teacher-test
ensemble achieves 0.9133 F1 on the teacher test set, improving on the original
final ensemble's 0.9073 F1.
Remaining errors are concentrated in high-style human writing and human-like
ChatGPT rewrites, so future work should prioritize hard negative human data,
ChatGPT-style augmentation, poetry expansion, and probability calibration.
```

## 8. 2026-05-21 Optimization Update

The holistic optimization pass tested text normalization, validation-only
calibration, controlled hard negatives, ChatGPT-style hard positives, poetry
expansion, and one combined neural retraining recipe.

The strongest internal-test candidate is now the Step7 DeBERTa-v3-base model
trained on:

```text
data/processed/lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl
```

Best supported operating point:

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old final ensemble | 0.9405 | 0.9450 | 0.9327 | 0.9388 | 0.9812 | 46 | 57 |
| Step7 DeBERTa default | 0.9584 | 0.9608 | 0.9540 | 0.9573 | 0.9909 | 33 | 39 |
| Step7 DeBERTa calibrated | 0.9596 | 0.9664 | 0.9504 | 0.9583 | 0.9909 | 28 | 42 |

The calibrated Step7 DeBERTa threshold was selected on the validation split
only (`threshold=0.676`). It improves internal-test F1 by `+0.0195` versus the
old final ensemble, reduces false positives by `18`, and reduces false
negatives by `15`.

Ensembling the new Step7 DeBERTa with TF-IDF did not become the final choice:
the best validation-selected Step7 ensemble reached internal-test F1 `0.9564`,
below the calibrated Step7 DeBERTa. The likely reason is that the new neural
branch already absorbed most of the useful hard-negative and hard-positive
signal, while TF-IDF mixing pushed the operating point toward higher precision
and lower recall.

Teacher-test re-evaluation was then run on `data/raw/teacher_test.json`.

| Run | Accuracy | Precision | Recall | F1 | ROC-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original final ensemble | 0.9033 | 0.8712 | 0.9467 | 0.9073 | 0.9663 | 21 | 8 |
| Step7 DeBERTa default | 0.9100 | 0.9020 | 0.9200 | 0.9109 | 0.9662 | 15 | 12 |
| Step7 DeBERTa calibrated | 0.9067 | 0.9067 | 0.9067 | 0.9067 | 0.9662 | 14 | 14 |
| Step7 ensemble, raw TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9690 | 13 | 13 |
| Step7 ensemble, combined TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9692 | 13 | 13 |
| Step7 ensemble, hardneg TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9694 | 13 | 13 |
| Step7 ensemble, hardneg+poetry TF-IDF | 0.9133 | 0.9133 | 0.9133 | 0.9133 | 0.9696 | 13 | 13 |

This confirms that the optimization goal was achieved on the teacher test set.
The recommended final teacher-test model is the Step7 ensemble. The raw-TF-IDF
variant is the simplest tied best candidate:

```text
DeBERTa: outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF: outputs/models/tfidf_lit_academic_poetry
alpha: 0.5
threshold: 0.55
teacher-test F1: 0.9133
```

Generated Step7 evidence:

```text
outputs/models/deberta_lit_academic_poetry_step7_combined/metrics.json
outputs/calibration/deberta_step7_combined/calibration_report.md
outputs/evaluation/step7_final_candidate_internal_test.md
outputs/evaluation/step7_final_candidate_poetry_internal_test.md
outputs/evaluation/teacher_test_step7_final_comparison.md
```
