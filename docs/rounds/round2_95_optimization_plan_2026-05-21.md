# Second-Round Optimization Plan For 95% Teacher-Test Accuracy

Updated: 2026-05-21

This document is the execution handoff for the next optimization round. The
goal is not another small threshold tweak. The goal is a new data, modeling,
fusion, and validation loop centered on one explicit target:

```text
95% teacher-test accuracy
```

Future conversations should read these files first:

```text
PROJECT_REPORT.md
docs/rounds/round2_95_optimization_plan_2026-05-21.md
```

Older detailed drafts and work logs are archived under:

```text
docs/rounds/
```

## 1. Current Baseline And Target Gap

The current recommended model is the Step7 DeBERTa + TF-IDF ensemble:

```text
DeBERTa:  outputs/models/deberta_lit_academic_poetry_step7_combined
TF-IDF:   outputs/models/tfidf_lit_academic_poetry
alpha:    0.5
threshold: 0.55
```

Teacher-test results:

| System | Accuracy | Correct | FP | FN | Confusion |
| --- | ---: | ---: | ---: | ---: | --- |
| Original final ensemble | 0.9033 | 271 / 300 | 21 | 8 | `[[129,21],[8,142]]` |
| Optimized Step7 ensemble | 0.9133 | 274 / 300 | 13 | 13 | `[[137,13],[13,137]]` |

The 95% target means:

```text
target correct = 285 / 300
current correct = 274 / 300
needed net gain = +11 correct examples
maximum allowed errors = 15
current errors = 26
```

Diagnostic ceiling from the existing model family:

| Diagnostic | Best observed teacher-test accuracy | Meaning |
| --- | ---: | --- |
| Existing single prediction file with oracle threshold | about 0.9267 | Threshold tuning alone is insufficient |
| Existing-family simple probability averaging | about 0.9333 | The current model family still falls short |
| Current Step7 final decision | 0.9133 | FP and FN are already balanced |

Therefore, the second round should not spend most of its effort on global
`threshold` or `alpha` tuning. Reaching 95% needs new information sources:

1. Data that is closer to the residual error distribution.
2. Domain-aware decision routing.
3. Nonlinear stacking fusion.
4. At least one third model branch with an error pattern different from
   DeBERTa and TF-IDF.

## 2. Evaluation Discipline

If the final report claims teacher-test generalization, these rules must hold:

1. Do not train on labels from `data/raw/teacher_test.json`.
2. Do not use teacher-test labels to choose thresholds, models, stackers, or
   routing parameters.
3. Use the teacher test only for final evaluation and post-hoc error
   interpretation.
4. Make all intermediate decisions using validation, internal test, and a new
   teacher-like development set.

If the instructor explicitly permits post-hoc repair against the known
teacher-test set, that can be built as a separate teacher-aware branch. It must
be clearly labeled in the report as a non-generalization result. The default
plan below assumes the strict generalization route.

## 3. Residual Error Profile

Current Step7 ensemble error IDs on teacher test:

```text
5, 32, 38, 57, 73, 101, 106, 107, 112, 120, 135, 141,
144, 152, 171, 181, 189, 197, 209, 246, 249, 264, 266,
285, 287, 292
```

Error types:

| Error type | Count | Main pattern |
| --- | ---: | --- |
| False positive | 13 | Human poetry, free verse, archaic verse, polished literary prose, formal academic prose |
| False negative | 13 | ChatGPT-like conservative rewrites, old-fiction style rewrites, archaic poem rewrites, natural academic paraphrases |

The model is not missing the task completely. Its boundary is unstable in a few
narrow regions:

1. High-style human writing is too often predicted as LLM.
2. Conservative ChatGPT-style rewrites are too often predicted as human.
3. Poetry and old-fiction style are the main conflict zones.
4. A single global threshold cannot reduce FP and FN at the same time.

## 4. Overall Second-Round Route

The second round has six phases.

| Phase | Goal | Main output |
| --- | --- | --- |
| 0. Baseline lock | Freeze diagnostics and the error ledger | `outputs/round2/error_ledger.*` |
| 1. Teacher-like dev set | Build a new development set without teacher-label leakage | `data/processed/round2_teacher_like_{train,dev}.jsonl` |
| 2. Domain router | Let poetry, academic text, and prose use different decision strategies | `outputs/models/round2_domain_router/` |
| 3. Stacking fusion | Replace linear alpha fusion | `outputs/models/round2_stacker/` |
| 4. Third branch | Add a heterogeneous neural model | `outputs/models/round2_roberta_or_electra/` |
| 5. Final selection | Compare strict candidates and pick the final system | `outputs/evaluation/round2_final_comparison.md` |

Recommended execution order:

```text
Phase 0 -> Phase 1 -> Phase 2 -> Phase 3
                       \-> Phase 4 -> Phase 5
```

Phase 2 and Phase 3 are low-cost, high-yield paths and should be done first.
Phase 4 costs more compute, but it is the most likely path to break the
diagnostic ceiling near 93%.

## 5. Phase 0: Error Ledger And Diagnostic Ceiling

### 5.1 Scripts To Add

```text
src/evaluation/export_error_ledger.py
src/evaluation/round2_threshold_family_diagnostics.py
```

`export_error_ledger.py` inputs:

```text
teacher/internal prediction jsonl
optional original input json/jsonl
```

Outputs:

```text
outputs/round2/error_ledger_teacher_step7.csv
outputs/round2/error_ledger_teacher_step7.jsonl
```

Suggested fields:

```text
id
label
prediction
error_type
probability
p_tfidf
p_deberta
text
length_chars
length_words
num_lines
linebreak_ratio
punctuation_ratio
quote_count
dash_count
archaic_word_count
academic_marker_count
rough_domain
confidence_bucket
notes
```

`round2_threshold_family_diagnostics.py` should take multiple prediction JSONL
files and write:

```text
outputs/round2/existing_family_threshold_report.md
outputs/round2/existing_family_threshold_report.json
```

The report should include:

1. Best threshold for each existing prediction file.
2. FP and FN for each prediction file.
3. Best diagnostic score for simple average ensembles.
4. Model error-overlap matrix.
5. Hard cases that every existing model family still misses.

### 5.2 Acceptance Criteria

After Phase 0, a later conversation should be able to answer:

1. What type each of the 26 residual teacher-test errors belongs to.
2. Which errors are threshold-repairable and which are not.
3. The diagnostic ceiling of the current model family.
4. Which samples require new data or a new model branch.

## 6. Phase 1: Teacher-Like Development Set

### 6.1 Core Principle

Do not expand data generically. Round 2 data must target residual errors:

1. Human hard negatives: teach the model that high-style human text is not LLM.
2. LLM hard positives: teach the model that conservative rewrites, archaic
   rewrites, and natural academic paraphrases are still LLM.
3. Every new sample should include `domain`, `subdomain`, `source`,
   `generator`, `pair_id`, and `round2_tag`.

### 6.2 Human Hard Negatives

Target: add `1,500-2,500` human-only hard negatives.

| Bucket | Target count | Purpose |
| --- | ---: | --- |
| classical_poetry_human | 400-600 | Repair classical-poetry false positives |
| modern_freeverse_human | 300-500 | Repair short free-verse false positives |
| ornate_literary_prose_human | 300-500 | Repair polished-prose false positives |
| short_fragment_human | 200-300 | Repair short-text overprediction as LLM |
| formal_academic_human | 300-600 | Repair academic-definition false positives |

Candidate sources:

```text
data/raw/external_human/poetry/
data/raw/external_human/gutenberg/
data/raw/external_human/academic/
```

If local sources are insufficient, prioritize public-domain literature and
poetry. Do not add teacher-test texts or near duplicates of teacher-test texts
to training.

### 6.3 LLM Hard Positives

Target: add `1,500-3,000` LLM positives, with outputs that look as human as
possible.

| Bucket | Target count | Prompt style |
| --- | ---: | --- |
| chatgpt_conservative_literary | 500-800 | Preserve original sentence rhythm and narrative pace; make only light rewrites |
| chatgpt_old_fiction | 300-500 | Imitate nineteenth-century fiction and avoid modern template wording |
| chatgpt_archaic_poetry | 300-500 | Preserve archaic spelling, rhyme, and line breaks; avoid modern AI style |
| chatgpt_natural_academic | 400-700 | Naturalize academic explanations without making them template-like |
| non_chatgpt_conservative | 300-500 | Use DeepSeek, Gemini, or Doubao for the same conservative rewrite styles |

Quality-filtering suggestions:

| Check | Suggested rule |
| --- | --- |
| Minimum length | `>= 60` words for prose, `>= 4` lines for poetry |
| Length ratio | `0.65 <= rewrite/source <= 1.35` |
| Lexical Jaccard | Prose `0.30-0.90`; academic may allow up to `0.92` |
| Repetition | Reject obvious repeated phrases |
| Empty or apology text | Reject |
| Prompt leakage | Reject mentions of rewrite instructions, AI, or ChatGPT |

### 6.4 Data Outputs

Create:

```text
data/processed/round2_human_hardneg_seed.jsonl
data/processed/round2_llm_hardpos_seed.jsonl
data/processed/round2_teacher_like_train.jsonl
data/processed/round2_teacher_like_dev.jsonl
data/processed/round2_teacher_like_report.json
```

`round2_teacher_like_dev.jsonl` is for tuning the router, stacker, and domain
thresholds. It must not contain teacher-test samples or near duplicates.

### 6.5 Acceptance Criteria

Phase 1 is not "more samples is always better." It must satisfy:

1. `round2_teacher_like_dev` covers at least 5 hard buckets.
2. Human and LLM rows are roughly balanced.
3. Poetry, old-fiction, and academic samples are sufficiently represented.
4. A manual spot check of 50 samples looks teacher-test-like.
5. There is no obvious `id`, `pair_id`, or near-duplicate leakage from the
   original train, validation, or internal-test splits.

## 7. Phase 2: Domain Router And Bucket Thresholds

### 7.1 Motivation

The current global threshold has a clear conflict:

1. Raising the threshold reduces human false positives but increases LLM false
   negatives.
2. Lowering the threshold improves LLM recall but hurts human poetry and
   polished prose.
3. Poetry, academic writing, and old prose have different optimal boundaries.

Round 2 should therefore start with domain-aware routing, not another single
global threshold.

### 7.2 Scripts To Add

```text
src/evaluation/assign_text_bucket.py
src/evaluation/tune_bucket_thresholds.py
src/evaluation/predict_bucket_routed_ensemble.py
```

Suggested buckets:

| Bucket | Signals |
| --- | --- |
| poetry_classical | Many line breaks, rhyme-like short lines, archaic words |
| poetry_freeverse | Short lines, low sentence completeness, modern image-heavy diction |
| literary_old_prose | Long sentences, narrative names, archaic or ornate punctuation |
| literary_short_fragment | Short narrative paragraph, low length |
| academic_formal | Academic markers, citations, terminology patterns |
| general_prose | Fallback |

Start with a rule-based router. It is transparent and appropriate for a course
report. A TF-IDF domain classifier can be added later if needed.

### 7.3 Threshold Selection

Choose thresholds per bucket on validation + `round2_teacher_like_dev`:

```text
threshold_bucket = argmax F1, with constraints:
  precision >= bucket_min_precision
  recall >= bucket_min_recall
```

Initial bias:

| Bucket | Bias |
| --- | --- |
| poetry_classical | Be more conservative when predicting LLM to reduce human FP |
| poetry_freeverse | Be more conservative when predicting LLM to reduce human FP |
| academic_formal | Balance dev errors while avoiding academic human FP |
| literary_old_prose | Conservative, but do not destroy old-fiction LLM recall |
| general_prose | Keep Step7 default threshold |

### 7.4 Acceptance Criteria

Before moving on, bucket routing should satisfy:

1. Overall internal-test F1 does not materially regress from the Step7
   calibrated DeBERTa F1 of `0.9583`.
2. Poetry internal-test F1 is above the old `0.8767`, or at least preserved.
3. ChatGPT recall is no lower than the old ensemble value of `0.7884`.
4. Human hard-negative FP decreases on validation and round2 dev.

If teacher-test diagnostics improve but validation and round2 dev get worse,
the router should not be promoted.

## 8. Phase 3: Nonlinear Stacking Fusion

### 8.1 Motivation

Linear fusion has only two free parameters:

```text
alpha
threshold
```

It cannot express:

1. A high DeBERTa probability in poetry is not always LLM evidence.
2. A low TF-IDF probability in academic text is not always human evidence.
3. Short and long texts have different calibration behavior.
4. Disagreement between DeBERTa and TF-IDF is itself informative.

### 8.2 Scripts To Add

```text
src/models/train_stacking_fusion.py
src/evaluation/predict_stacking_fusion.py
src/evaluation/compare_round2_candidates.py
```

### 8.3 Stacker Features

First version: Logistic Regression. If needed, try
`HistGradientBoostingClassifier`.

Suggested features:

```text
p_tfidf
p_deberta_step7
p_deberta_old
p_ensemble_step7
abs(p_tfidf - p_deberta_step7)
max(p_tfidf, p_deberta_step7)
min(p_tfidf, p_deberta_step7)
length_chars
length_words
num_lines
linebreak_ratio
avg_line_length
punctuation_ratio
quote_count
dash_count
semicolon_count
archaic_word_count
academic_marker_count
type_token_ratio
sentence_length_mean
sentence_length_std
bucket_one_hot
```

The feature extraction can reuse Phase 0 error-ledger features.

### 8.4 Training Protocol

Most practical first version:

1. Freeze base models.
2. Train the stacker on validation + `round2_teacher_like_dev`.
3. Use internal test as the main pre-teacher-test evaluation.
4. Run teacher test only once at the end.
5. Use strong regularization to reduce small-meta-set overfitting.

Stricter version:

1. Generate K-fold out-of-fold base predictions on the train split.
2. Train the stacker on OOF predictions.
3. Select parameters on validation.
4. Treat internal test as holdout.

If time is limited, start with the first version and state clearly in the
report that it is lightweight stacking.

### 8.5 Acceptance Criteria

| Metric | Minimum requirement |
| --- | --- |
| internal_test F1 | `>= 0.9600` |
| internal_test FP | `<= 28-33` |
| internal_test FN | `<= 37-42` |
| ChatGPT recall | Preferred `>= 0.84`, minimum `>= 0.80` |
| poetry F1 | Preferred `>= 0.89`, minimum no drop |
| round2_dev hard buckets | Clear improvement over Step7 ensemble |

If the stacker improves only teacher-test diagnostics while internal test and
round2 dev do not improve, treat it as overfitting and do not promote it.

## 9. Phase 4: Third Model Branch

### 9.1 Motivation

DeBERTa and TF-IDF already complement each other, but their remaining errors
still overlap. To break the ceiling near 93%, add a branch with a different
error pattern.

### 9.2 Candidate Models

Ordered by cost:

| Candidate | Reason | Cost |
| --- | --- | --- |
| `roberta-base` | Different tokenizer and pretraining; manageable training cost | medium |
| `google/electra-base-discriminator` | Discriminative pretraining may fit detection | medium |
| `roberta-large` | Higher capacity but more memory pressure | high |
| `microsoft/deberta-v3-large` | Higher capacity but more correlated with current DeBERTa | high |

Recommended order:

```text
1. roberta-base
2. google/electra-base-discriminator
3. roberta-large if hardware allows
```

### 9.3 Training Command Templates

`src/models/train_deberta.py` uses `AutoModelForSequenceClassification`, so
the same script can train these branches by changing `--model_name`.

RoBERTa branch:

```powershell
python src/models/train_deberta.py `
  --train data/processed/round2_teacher_like_train.jsonl `
  --valid data/processed/lit_academic_poetry_valid.jsonl `
  --test data/processed/lit_academic_poetry_internal_test.jsonl `
  --output_dir outputs/models/round2_roberta_base `
  --model_name roberta-base `
  --max_length 512 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --learning_rate 1e-5 `
  --epochs 3
```

ELECTRA branch:

```powershell
python src/models/train_deberta.py `
  --train data/processed/round2_teacher_like_train.jsonl `
  --valid data/processed/lit_academic_poetry_valid.jsonl `
  --test data/processed/lit_academic_poetry_internal_test.jsonl `
  --output_dir outputs/models/round2_electra_base `
  --model_name google/electra-base-discriminator `
  --max_length 512 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --learning_rate 1e-5 `
  --epochs 3
```

If GPU memory is insufficient:

```text
reduce batch_size to 2
increase gradient_accumulation_steps to 4
enable fp16 if CUDA supports it
```

### 9.4 Promotion Criteria

The third model is not useful only because of standalone F1. It must make
different mistakes.

Promotion criteria:

1. Internal-test F1 is close to or above Step7 DeBERTa.
2. It clearly repairs at least one hard bucket on teacher-like dev.
3. Error overlap with Step7 ensemble is below `75%`.
4. Adding it to the stacker improves internal test and round2 dev.
5. It does not collapse poetry performance or ChatGPT recall.

If the third model has high F1 but the same errors as DeBERTa, its value is
limited.

## 10. Phase 5: Final Candidate Comparison

### 10.1 Candidate Set

Compare at least:

| Candidate | Description |
| --- | --- |
| step7_ensemble | Current final model, kept as baseline |
| bucket_routed_step7 | Step7 + domain router + bucket thresholds |
| stacker_step7 | TF-IDF + DeBERTa + text-feature stacker |
| roberta_single | Third branch standalone |
| stacker_with_roberta | Step7 + RoBERTa/ELECTRA + text features |
| precision_guard_variant | More conservative on false positives |
| recall_guard_variant | More conservative on false negatives |

### 10.2 Report Outputs

Write:

```text
outputs/evaluation/round2_internal_comparison.md
outputs/evaluation/round2_teacher_like_dev_comparison.md
outputs/evaluation/round2_final_teacher_comparison.md
outputs/evaluation/round2_error_overlap_matrix.csv
outputs/predictions/round2_final_submission.json
```

### 10.3 Final Promotion Gates

Strict-route candidate requirements:

| Gate | Requirement |
| --- | --- |
| validation | No worse than current Step7-related candidates |
| internal_test | Ideally `>= 0.963`, minimum `>= 0.960` |
| round2_teacher_like_dev | Clear improvement on hard buckets over Step7 |
| poetry | No worse than current poetry F1, preferably better |
| ChatGPT recall | Higher than current `0.7884`, target `>= 0.84` |
| FP/FN balance | No extreme tradeoff that only fixes one side |
| teacher-test final | Target `>= 285/300` |

If teacher-test accuracy remains below 93%, do not keep blindly tuning. Return
to Phase 1 for more teacher-like data or to Phase 4 for a different model
branch.

## 11. Concrete Work Blocks

### Block A: Diagnostics And Ledger

Deliverables:

```text
src/evaluation/export_error_ledger.py
src/evaluation/round2_threshold_family_diagnostics.py
outputs/round2/error_ledger_teacher_step7.csv
outputs/round2/existing_family_threshold_report.md
```

Commands:

```powershell
python src/evaluation/export_error_ledger.py `
  --predictions outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl `
  --input data/raw/teacher_test.json `
  --output_csv outputs/round2/error_ledger_teacher_step7.csv `
  --output_jsonl outputs/round2/error_ledger_teacher_step7.jsonl

python src/evaluation/round2_threshold_family_diagnostics.py `
  --predictions outputs/predictions/teacher_test_*predictions.jsonl `
  --output_md outputs/round2/existing_family_threshold_report.md `
  --output_json outputs/round2/existing_family_threshold_report.json
```

### Block B: Round2 Teacher-Like Data

Deliverables:

```text
src/data/build_round2_teacher_like_set.py
src/data/prepare_round2_hard_positive_prompts.py
data/processed/round2_human_hardneg_seed.jsonl
data/processed/round2_llm_hardpos_seed.jsonl
data/processed/round2_teacher_like_train.jsonl
data/processed/round2_teacher_like_dev.jsonl
data/processed/round2_teacher_like_report.json
```

Acceptance:

```text
hard buckets covered: >= 5
round2 dev rows: >= 800
round2 train additions: >= 2500
human/LLM balance: no class below 40%
manual spot check: 50 samples
```

### Block C: Domain Router

Deliverables:

```text
src/evaluation/assign_text_bucket.py
src/evaluation/tune_bucket_thresholds.py
src/evaluation/predict_bucket_routed_ensemble.py
outputs/models/round2_bucket_thresholds.json
outputs/evaluation/round2_bucket_router_report.md
```

Acceptance:

```text
round2 dev hard-negative FP decreases
internal_test F1 does not regress materially
poetry F1 improves or is preserved
ChatGPT recall is preserved
```

### Block D: Stacking Fusion

Deliverables:

```text
src/models/train_stacking_fusion.py
src/evaluation/predict_stacking_fusion.py
outputs/models/round2_stacker/
outputs/evaluation/round2_stacker_report.md
```

Acceptance:

```text
internal_test F1 >= 0.960
round2 dev hard buckets improve
error overlap with Step7 decreases
```

### Block E: Third Branch

Deliverables:

```text
outputs/models/round2_roberta_base/
outputs/models/round2_electra_base/
outputs/evaluation/round2_third_branch_report.md
```

Acceptance:

```text
third branch has non-identical errors
stacker_with_third_branch beats stacker_step7 on internal_test or round2_dev
```

### Block F: Final Comparison And Handoff

Deliverables:

```text
outputs/evaluation/round2_final_comparison.md
outputs/predictions/round2_final_teacher_predictions.jsonl
outputs/predictions/round2_final_submission.json
docs/rounds/round2_results_summary_2026-05-22.md
```

Acceptance:

```text
teacher-test target: >= 285 / 300
if below target: explain which error buckets remain and what data/model would be needed
```

## 12. Risk And Fallback Plan

| Risk | Symptom | Fallback |
| --- | --- | --- |
| Router overfits buckets | round2 dev improves, internal test drops | Keep router only as a stacker feature |
| New hard negatives reduce recall | FP drops, FN rises | Lower hard-negative weight and add paired hard positives |
| ChatGPT positives are too artificial | Internal ChatGPT recall does not move | Rewrite prompts to be more conservative and old-style |
| Third model duplicates DeBERTa errors | Same mistakes, no ensemble gain | Try ELECTRA or a stylometry-only branch |
| Stacker overfits small meta set | Teacher diagnostics improve only | Use stronger regularization or OOF predictions |
| 95% still unreachable | Strict metrics plateau near 93% | Report signal limits and add more teacher-like data |

## 13. Minimum Viable Second Round

If time is limited, execute in this order:

1. Build `export_error_ledger.py` and residual-error typing.
2. Build `round2_teacher_like_dev` with at least 800 rows.
3. Build bucket router + per-bucket thresholds.
4. Build Logistic Regression stacker.
5. Train `roberta-base` as the third branch.
6. Use `stacker_with_roberta` as the final candidate.

Success signal:

```text
internal_test F1 >= 0.960
round2_teacher_like_dev hard buckets clearly improve
teacher-test correct >= 285 / 300
```

If the first four steps already push teacher-test diagnostics to around 94%,
then train the third branch. If they remain below 93%, return to data
construction before spending time on more threshold tuning.

## 14. Suggested Prompt For A Future Conversation

```text
Please read PROJECT_REPORT.md and docs/rounds/round2_95_optimization_plan_2026-05-21.md.
The goal is to improve the current Step7 ensemble from 274/300 to at least
285/300 on the teacher test. Start from Phase 0 by implementing the error
ledger and threshold-family diagnostics, then proceed through round2
teacher-like dev data, domain router, stacking fusion, and the third model
branch.
```
