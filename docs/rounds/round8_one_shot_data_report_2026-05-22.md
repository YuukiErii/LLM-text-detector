# Round8-OneShot Data Report

Updated: 2026-05-22

This report records the completed data-preparation phases for the final
Round8-OneShot residual optimization route.

Teacher-test boundary:

```text
data/raw/teacher_test.json was used only for exact-text duplicate exclusion.
It was not used for training, threshold tuning, selector calibration, rule
search, model selection, or per-row decisions.
```

## 1. Taxonomy

Created:

```text
docs/rounds/round8_residual_error_taxonomy_2026-05-22.md
```

It defines:

```text
8 human hard-negative buckets
8 LLM hard-positive buckets
required metadata
candidate-pool policy
hard residual selection policy
group split and leakage policy
non-teacher promotion gates
```

## 2. Residual Candidate Pool

Script:

```text
src/data/build_residual_candidate_pool.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\data\build_residual_candidate_pool.py
```

Outputs:

```text
data/processed/residual_candidate_pool_v1.jsonl
data/processed/residual_candidate_pool_v1_report.json
```

Summary:

| Item | Value |
| --- | ---: |
| total candidates | 7,651 |
| human hard negatives | 4,040 |
| LLM hard positives | 3,611 |
| split groups | 5,224 |
| minimum label share | 47.20% |
| taxonomy buckets present | 16 / 16 |
| teacher-test exact duplicates | 0 |
| ready for Step7 scoring | yes |

Source policy:

```text
Default inputs are the already validated Round4 residual seeds:
data/processed/round4_hard_human_mirror_seed.jsonl
data/processed/round4_hard_llm_positive_seed.jsonl
```

The script remaps Round4 metadata into Round8 taxonomy fields instead of using
old Round4 bucket names as final labels.

## 3. Step7 Scoring

Script:

```text
src/evaluation/predict_step7_on_residual_candidates.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\evaluation\predict_step7_on_residual_candidates.py --batch_size 32
```

Outputs:

```text
outputs/predictions/residual_candidate_pool_v1_step7_predictions.jsonl
outputs/predictions/residual_candidate_pool_v1_step7_metrics.json
```

Frozen Step7 config:

```text
TF-IDF:  outputs/models/tfidf_lit_academic_poetry
DeBERTa: outputs/models/deberta_lit_academic_poetry_step7_combined
alpha:   0.5
threshold: 0.55
```

Overall residual-pool metrics:

| Accuracy | Precision | Recall | F1 | FP | FN |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8544 | 0.9664 | 0.7164 | 0.8228 | 90 | 1,024 |

Confusion matrix:

```text
[[3950, 90], [1024, 2587]]
```

Hard residual counts:

| Category | Rows |
| --- | ---: |
| Step7 errors | 1,114 |
| ambiguous zone | 407 |
| hard human candidates | 90 |
| very hard human candidates | 53 |
| hard LLM candidates | 957 |
| very hard LLM candidates | 858 |

Interpretation:

```text
The pool strongly exposes Step7's residual false-negative region while keeping
human hard-negative FP limited. Residual training therefore needs explicit human
support rows to preserve precision, not only Step7-misclassified human rows.
```

## 4. Hard Residual Split

Script:

```text
src/data/select_step7_hard_residuals.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\data\select_step7_hard_residuals.py
```

Outputs:

```text
data/processed/residual_selected_v1.jsonl
data/processed/residual_train_v1.jsonl
data/processed/residual_dev_v1.jsonl
data/processed/residual_probe_v1.jsonl
data/processed/residual_split_v1_report.json
```

Selection policy:

```text
Core hard human: label=0 and p_step7>=0.55
Core hard LLM:   label=1 and p_step7<=0.45
Ambiguous zone:  0.35<=p_step7<=0.65
Support rows:    filled by hardness score from the same non-teacher pool
```

Split summary:

| Split | Rows | Human | LLM | Step7 Error Rows | Core Hard Rows | Ambiguous Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| residual_train_v1 | 3,064 | 1,264 | 1,800 | 523 | 463 | 320 |
| residual_dev_v1 | 800 | 400 | 400 | 370 | 367 | 28 |
| residual_probe_v1 | 426 | 226 | 200 | 221 | 217 | 59 |

Leakage checks:

| Check | Count |
| --- | ---: |
| train/dev group overlap | 0 |
| train/probe group overlap | 0 |
| dev/probe group overlap | 0 |
| train/dev text overlap | 0 |
| train/probe text overlap | 0 |
| dev/probe text overlap | 0 |
| teacher-test exact text duplicates | 0 |

Decision:

```text
ready_for_residual_mix_train_build = yes
```

## 5. Round8 Residual Mix Train

Script:

```text
src/data/build_round8_residual_mix_train.py
```

Command:

```powershell
.\.venv\Scripts\python.exe src\data\build_round8_residual_mix_train.py
```

Outputs:

```text
data/processed/lit_academic_poetry_train_round8_residual_mix.jsonl
data/processed/lit_academic_poetry_train_round8_residual_mix_report.json
```

Base train:

```text
data/processed/lit_academic_poetry_train_hardneg_p50_chatgpt_hardpos_poetry_expansion.jsonl
```

Mix result:

| Item | Value |
| --- | ---: |
| total rows | 10,213 |
| base rows | 7,149 |
| residual rows | 3,064 |
| residual ratio | 0.3000 |
| ready for DeBERTa training | yes |

The script gives residual rows priority during exact-text deduplication, then
backfills base rows to keep the requested 70/30 ratio.

## 6. Next Step

The next phase is residual-aware DeBERTa fine-tuning from the current Step7
checkpoint.

Suggested first training command:

```powershell
.\.venv\Scripts\python.exe src\models\train_deberta.py `
  --train data\processed\lit_academic_poetry_train_round8_residual_mix.jsonl `
  --valid data\processed\lit_academic_poetry_valid.jsonl `
  --test data\processed\lit_academic_poetry_internal_test.jsonl `
  --output_dir outputs\models\deberta_round8_residual_mix `
  --model_name outputs\models\deberta_lit_academic_poetry_step7_combined\best_model `
  --max_length 512 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --learning_rate 5e-6 `
  --epochs 2 `
  --weight_decay 0.01 `
  --warmup_ratio 0.1
```

Before any teacher-test diagnostic, the trained candidate must be evaluated on:

```text
original valid
original internal_test
residual_dev_v1
residual_probe_v1
hard human negatives
hard LLM positives
domain and generator breakdowns
```
