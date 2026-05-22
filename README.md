# LLM Text Detector

Hybrid DeBERTa-TFIDF detector for an NLG course project. The task is to decide
whether an English passage is human-written or generated / rewritten by an LLM.

The project focuses on rewritten text rather than generic AI essay detection.
The expected test distribution includes literary prose, archaic English, poetry,
and NLP / computational linguistics academic paragraphs.

## Task

Input:

```json
{"text": "The passage to classify..."}
```

Output:

```json
{"label": 1, "probability": 0.83}
```

Labels:

| Label | Meaning |
| --- | --- |
| `0` | Human-written text |
| `1` | LLM-generated or LLM-rewritten text |

## Model Plan

The original delivered detector was a probability ensemble:

```text
P_final = alpha * P_deberta + (1 - alpha) * P_tfidf
```

Branches:

| Branch | Role |
| --- | --- |
| DeBERTa-v3-base classifier | Captures semantic, discourse, and deep style signals |
| Word/char TF-IDF + Logistic Regression | Captures lexical, punctuation, spelling, and surface style signals |

After the 2026-05-21 optimization pass, the strongest teacher-test candidate is
the Step7 DeBERTa-v3-base model retrained on the combined hard-negative,
ChatGPT-hard-positive, and poetry-expansion recipe, then ensembled with TF-IDF.
The best validated teacher-test variants reach F1 `0.9133` versus `0.9073` for
the original final ensemble.

The 2026-05-22 Round3 pass added an OOF stacker and precision-guarded routing.
Those candidates improved the constructed guard development set, but did not
beat Step7 on the final teacher-test diagnostic. The final public system
therefore remains the Step7 DeBERTa + TF-IDF ensemble.

Later 2026-05-22 Round4-Round6 work explored residual-repair candidates:
paired residual data, a human-style guard, a flip guard, and a safe-override
selector. These runs produced useful code and diagnostics, but none cleared the
promotion gates for replacing Step7. The current next route is Round7:
exact-candidate calibrated selector work, documented in
`docs/ROUND6_DETAILED_WORK_RECORD_AND_ROUND7_PLAN_2026-05-22.md`.

TF-IDF, DeBERTa, threshold calibration, comparison, and probability ensemble
entrypoints are present. Training artifacts are intentionally not committed.

## Current Status

Current processed data includes:

| Artifact | Status |
| --- | --- |
| `data/processed/human_seed.jsonl` | Literature human seed, 7130 samples |
| `data/processed/academic_seed.jsonl` | Academic human seed, 1200 samples |
| `data/processed/poetry_seed.jsonl` | Poetry human seed, 500 samples |
| `data/processed/human_seed_combined.jsonl` | Combined human seed, 8830 samples |
| `data/processed/rewrite_prompts.jsonl` | Literature rewrite prompts, 7130 tasks |
| `data/processed/rewrite_prompts_academic.jsonl` | Academic rewrite prompts, 1098 tasks |
| `data/processed/rewrite_prompts_poetry.jsonl` | Poetry rewrite prompts, 500 tasks |
| Literature LLM rewrites | DeepSeek, Doubao, Gemini, and ChatGPT branches generated |
| Academic LLM rewrites | ChatGPT, DeepSeek, Gemini, and Doubao branches generated |
| Poetry LLM rewrites | ChatGPT, DeepSeek, Gemini, and Doubao branches generated |
| Gemini literature rerun | Merged into `llm_rewrite_gemini_clean.jsonl` |
| `data/processed/full_dataset_lit_academic_poetry.jsonl` | Main full dataset, 17295 samples |
| `data/processed/lit_academic_poetry_{train,valid,internal_test}.jsonl` | Main pair-safe split |
| TF-IDF baseline | Implemented in `src/models/train_tfidf_baseline.py` |
| DeBERTa classifier | Implemented in `src/models/train_deberta.py` |
| Ensemble tuner | Implemented in `src/models/ensemble.py`; original fine config uses `alpha=0.33`, `threshold=0.48` |
| Threshold calibration | Implemented in `src/evaluation/calibrate_prediction_thresholds.py`; Step7 DeBERTa internal-test best-F1 threshold is `0.676` |
| Final ensemble inference | Implemented in `src/evaluation/predict_ensemble.py` |
| API | Optional future serving work |

For detailed handoff notes, read `PROJECT_REPORT.md` first. For the second
round 95% target, read `docs/SECOND_ROUND_95_OPTIMIZATION_PLAN.md`; for the
executed second-round result, read `docs/ROUND2_RESULTS_SUMMARY.md`; for the
third-round OOF and precision-guard result, read
`docs/ROUND3_RESULTS_SUMMARY.md`. For the later residual-repair sequence, read
`docs/ROUND4_V1_SUMMARY_AND_ROUND5_PLAN_2026-05-22.md`,
`docs/ROUND5_FINAL_DECISION_2026-05-22.md`, and
`docs/ROUND6_DETAILED_WORK_RECORD_AND_ROUND7_PLAN_2026-05-22.md`. Older drafts,
plans, and full work logs are preserved in `docs/archive/`.

## Repository Layout

```text
.
|-- data/
|   |-- raw/                 # Raw external caches ignored, teacher_test.json tracked
|   `-- processed/           # Tracked processed JSONL datasets and reports
|-- outputs/
|   |-- figures/             # Ignored generated figures
|   |-- models/              # Ignored trained artifacts
|   `-- predictions/         # Ignored prediction outputs
|-- docs/
|   |-- SECOND_ROUND_95_OPTIMIZATION_PLAN.md
|   |-- ROUND2_POSTMORTEM_AND_ROUND3_PLAN.md
|   |-- ROUND3_RESULTS_SUMMARY.md
|   |-- ROUND4_V1_SUMMARY_AND_ROUND5_PLAN_2026-05-22.md
|   |-- ROUND5_FINAL_DECISION_2026-05-22.md
|   |-- ROUND6_DETAILED_WORK_RECORD_AND_ROUND7_PLAN_2026-05-22.md
|   `-- archive/             # Preserved historical markdown drafts and logs
|-- src/
|   |-- app/                 # Future FastAPI app
|   |-- data/                # Data collection, prompt, rewrite, QC, split scripts
|   |-- evaluation/          # Final inference and prediction analysis scripts
|   |-- features/            # Future feature utilities
|   `-- models/              # Baseline and future neural training scripts
|-- requirements.txt
|-- PROJECT_REPORT.md
`-- README.md
```

`data/processed/` and `data/raw/teacher_test.json` are tracked so the project
can be reproduced from the repository. Large raw external caches under
`data/raw/external_*`, `outputs/`, `.env`, `.venv/`, and local IDE files are
intentionally ignored by Git.

## Setup

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is not on PATH, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file for rewrite generation. Do not commit it.

```text
GPTSAPI_API_KEY=your_gptsapi_api_key
DEEPSEEK_API_KEY=...
ARK_API_KEY=...

# Optional model overrides
CHATGPT_MODEL=gpt-5.4-mini
GEMINI_MODEL=gemini-3-flash-preview
DOUBAO_MODEL=doubao-seed-2-0-pro-260215
```

The GPTSAPI key is used by the ChatGPT and Gemini rewrite scripts. The ARK key
is used by Doubao.

## Common Commands

Inspect human seed:

```powershell
python src/data/inspect_human_seed.py
```

Inspect LLM rewrite quality:

```powershell
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_chatgpt.jsonl
python src/data/inspect_llm_rewrite.py --input data/processed/llm_rewrite_academic_deepseek.jsonl
```

Prepare Gemini literature rerun prompts:

```powershell
python src/data/prepare_gemini_literature_rerun_prompts.py
```

Run Gemini literature rerun:

```powershell
python src/data/generate_gemini_rewrites.py `
  --input data/processed/rewrite_prompts_gemini_literature_rerun.jsonl `
  --output data/processed/llm_rewrite_gemini_rerun.jsonl `
  --failed data/processed/llm_rewrite_gemini_rerun_failed.jsonl `
  --limit -1 `
  --model gemini-3-flash-preview `
  --max_tokens 3000 `
  --temperature 0.4 `
  --top_p 0.8 `
  --sleep 0.5
```

Merge Gemini rerun outputs into a clean literature file:

```powershell
python src/data/merge_gemini_literature_rerun.py
```

Build a full dataset:

```powershell
python src/data/build_full_dataset.py `
  --human data/processed/human_seed.jsonl `
  --llm `
    data/processed/llm_rewrite_deepseek.jsonl `
    data/processed/llm_rewrite_doubao.jsonl `
    data/processed/llm_rewrite_chatgpt.jsonl `
    data/processed/llm_rewrite_gemini_clean.jsonl `
  --output data/processed/full_dataset_literature.jsonl `
  --report data/processed/full_dataset_literature_report.json
```

Build the final literature + academic + poetry dataset:

```powershell
python src/data/merge_human_seeds.py `
  --inputs `
    data/processed/human_seed.jsonl `
    data/processed/academic_seed.jsonl `
    data/processed/poetry_seed.jsonl `
  --output data/processed/human_seed_combined.jsonl `
  --report data/processed/human_seed_combined_report.json

python src/data/build_full_dataset.py `
  --human data/processed/human_seed_combined.jsonl `
  --llm `
    data/processed/llm_rewrite_deepseek.jsonl `
    data/processed/llm_rewrite_doubao.jsonl `
    data/processed/llm_rewrite_chatgpt.jsonl `
    data/processed/llm_rewrite_gemini_clean.jsonl `
    data/processed/llm_rewrite_academic_chatgpt.jsonl `
    data/processed/llm_rewrite_academic_deepseek.jsonl `
    data/processed/llm_rewrite_academic_gemini.jsonl `
    data/processed/llm_rewrite_academic_doubao.jsonl `
    data/processed/llm_rewrite_poetry_chatgpt.jsonl `
    data/processed/llm_rewrite_poetry_deepseek.jsonl `
    data/processed/llm_rewrite_poetry_gemini.jsonl `
    data/processed/llm_rewrite_poetry_doubao.jsonl `
  --output data/processed/full_dataset_lit_academic_poetry.jsonl `
  --report data/processed/full_dataset_lit_academic_poetry_report.json
```

Split by `pair_id`:

```powershell
python src/data/split_dataset_by_pair.py `
  --input data/processed/full_dataset_lit_academic_poetry.jsonl `
  --prefix lit_academic_poetry_
```

Train TF-IDF baseline:

```powershell
python src/models/train_tfidf_baseline.py `
  --train data/processed/lit_academic_poetry_train.jsonl `
  --valid data/processed/lit_academic_poetry_valid.jsonl `
  --test data/processed/lit_academic_poetry_internal_test.jsonl `
  --output_dir outputs/models/tfidf_lit_academic_poetry
```

Fine-tune DeBERTa:

```powershell
python src/models/train_deberta.py `
  --train data/processed/lit_academic_poetry_train.jsonl `
  --valid data/processed/lit_academic_poetry_valid.jsonl `
  --test data/processed/lit_academic_poetry_internal_test.jsonl `
  --output_dir outputs/models/deberta_lit_academic_poetry `
  --model_name microsoft/deberta-v3-base `
  --max_length 512 `
  --batch_size 4 `
  --eval_batch_size 8 `
  --gradient_accumulation_steps 2 `
  --learning_rate 1e-5 `
  --epochs 3
```

Tune DeBERTa + TF-IDF ensemble after both prediction files exist:

```powershell
python src/models/ensemble.py `
  --valid_deberta outputs/models/deberta_lit_academic_poetry/predictions/deberta_valid_predictions.jsonl `
  --valid_tfidf outputs/models/tfidf_lit_academic_poetry/predictions/tfidf_valid_predictions.jsonl `
  --test_deberta outputs/models/deberta_lit_academic_poetry/predictions/deberta_internal_test_predictions.jsonl `
  --test_tfidf outputs/models/tfidf_lit_academic_poetry/predictions/tfidf_internal_test_predictions.jsonl `
  --output_dir outputs/models/ensemble_lit_academic_poetry_fine `
  --alphas 0.25,0.26,0.27,0.28,0.29,0.3,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55 `
  --thresholds 0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6
```

Run the optimized Step7 ensemble inference on the teacher test set:

```powershell
python src/evaluation/predict_ensemble.py `
  --input data/raw/teacher_test.json `
  --output outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl `
  --submission outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_submission.json `
  --metrics outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_metrics.json `
  --tfidf_dir outputs/models/tfidf_lit_academic_poetry `
  --deberta_dir outputs/models/deberta_lit_academic_poetry_step7_combined `
  --alpha 0.5 `
  --threshold 0.55 `
  --batch_size 16 `
  --minimal_submission
```

The original final ensemble configuration is loaded from:

```text
outputs/models/ensemble_lit_academic_poetry_fine/fusion_config.json
```

Current final teacher-test result:

```text
final system = Step7 DeBERTa + TF-IDF ensemble
accuracy  0.9133
precision 0.9133
recall    0.9133
f1        0.9133
roc_auc   0.9690
confusion [[137, 13], [13, 137]]
```

The previous final ensemble remains a useful baseline:

```text
accuracy  0.9033
precision 0.8712
recall    0.9467
f1        0.9073
roc_auc   0.9663
confusion [[129, 21], [8, 142]]
```

The minimal submission file is a JSON list with one record per input sample:

```json
{"label": 1, "probability": 0.83}
```

Generate a teacher-test error-analysis note:

```powershell
python src/evaluation/analyze_predictions.py `
  --predictions outputs/predictions/teacher_test_step7_ensemble_raw_tfidf_predictions.jsonl `
  --input data/raw/teacher_test.json `
  --output outputs/predictions/teacher_test_step7_error_analysis.md `
  --threshold 0.55 `
  --examples 8
```

Report-ready generated artifacts:

```text
outputs/predictions/final_report_tables.md
outputs/predictions/teacher_test_step7_error_analysis.md
outputs/predictions/round3_final_submission.json
outputs/evaluation/round3_final_teacher_comparison.md
```

The public repository does not commit `outputs/` files. The report-ready
metrics and the detailed optimization roadmap are consolidated in:

```text
PROJECT_REPORT.md
```

## Data Rules

- Do not use the teacher test JSON for training, threshold tuning, model
  selection, or prompt selection.
- Keep each human source and its LLM rewrites in the same split via `pair_id`.
- Prefer quality filtering over maximizing sample count.
- Keep generator and domain metadata for ablation and error analysis.

## Next Work Items

1. Use `PROJECT_REPORT.md` as the single report-facing summary.
2. If further accuracy is needed, continue from the Round7 route in
   `docs/ROUND6_DETAILED_WORK_RECORD_AND_ROUND7_PLAN_2026-05-22.md`: audit
   exact override candidates, build an exact-candidate dataset, and promote to
   teacher-test only after non-teacher gates pass.
3. Add API serving only if required by the final submission format.
