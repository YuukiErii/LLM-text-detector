# Project File Manifest

Updated: 2026-05-22

This manifest records the cleaned repository surface after final project
organization.

## Public Repository Surface

| Path | Status | Purpose |
| --- | --- | --- |
| `README.md` | tracked | setup, overview, common commands |
| `PROJECT_REPORT.md` | tracked | consolidated English project handoff |
| `requirements.txt` | tracked | Python dependencies |
| `src/` | tracked | data, model, and evaluation scripts |
| `data/raw/teacher_test.json` | tracked | final diagnostic input supplied for the project |
| `data/processed/` | tracked | processed reproducibility datasets and reports |
| `docs/rounds/` | tracked | normalized round-by-round plans, logs, and decisions |
| `docs/final/` | tracked | final model card and file manifest |
| `outputs/models/deberta_lit_academic_poetry_step7_combined/` | tracked via LFS for weights | final Step7 DeBERTa inference artifact |
| `outputs/models/tfidf_lit_academic_poetry/` | tracked | final TF-IDF branch |
| `outputs/models/ensemble_lit_academic_poetry_step7_deberta_raw_tfidf/` | tracked | final fusion config and small validation/internal predictions |
| `outputs/models/ensemble_lit_academic_poetry_fine/` | tracked | original final ensemble config for comparison |
| `outputs/figures/publication/` | tracked | final publication-style figures |

## Local-Only Surface

| Path / pattern | Reason |
| --- | --- |
| `report/` | Chinese final report source/build products; course deliverable, not public repo content |
| `presentation/` | English final presentation; course deliverable, not public repo content |
| `data/raw/external_*` | raw external caches, large and reproducible from source scripts |
| `outputs/predictions/` | generated prediction dumps |
| `outputs/evaluation/` | generated comparison and tuning outputs |
| `outputs/calibration/` | generated calibration sweeps |
| non-final `outputs/models/*` | non-promoted checkpoints, guards, stackers, and selectors |
| `.env`, `.env.*` | secrets and local API credentials |
| `.venv/`, IDE files, caches | local environment files |

## Naming Conventions

Documentation now follows lowercase snake_case filenames:

```text
docs/rounds/round{n}_{topic}_{date}.md
docs/final/{artifact_name}.md
```

Model artifact names are intentionally left in their original training-output
form when scripts or reports refer to them directly. Only final Step7 inference
artifacts remain under `outputs/models/`.

## Cleanup Decision

Non-final heavyweight model directories and raw external caches were removed
from the project tree after confirming that Step7 is the final baseline. Round
records, reports, metrics, and code paths were preserved so the optimization
history remains auditable without carrying non-promoted weights.
