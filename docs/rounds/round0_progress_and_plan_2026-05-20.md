# LLM Text Detector Progress And Plan

Archived: 2026-05-21

This archived document is the English version of the 2026-05-20 progress
handoff. It records the project state immediately before the full
literature + academic + poetry dataset and the final Step7 optimization pass
were completed.

## 1. Project State On 2026-05-20

The project had moved from a literature-only prototype toward a multi-domain
detector for:

```text
label = 0: human-written original text
label = 1: LLM-generated or LLM-rewritten text
```

The target distribution was understood to include:

1. Nineteenth- and twentieth-century English fiction.
2. Archaic or classical English.
3. Poetry and strongly formatted text.
4. NLP / computational linguistics academic paragraphs.
5. LLM rewrites of those styles.

The planned final detector remained:

```text
Hybrid DeBERTa-TFIDF Detector

P_final = alpha * P_deberta + (1 - alpha) * P_tfidf
```

## 2. Completed Data Work

By this point, the project had already completed or started:

| Item | Status |
| --- | --- |
| Gutenberg literature human seed | Built and usable |
| Literature rewrite prompts | Built and split by generator |
| DeepSeek literature rewrites | Completed and usable |
| Gemini literature rewrites | Repaired after truncation issues |
| ChatGPT literature rewrites | Running with improved parameters |
| ACL-OCL academic route | Validated |
| Academic human seed | Built with 1,200 samples |
| Academic prompts | Filtered to 1,098 high-quality tasks |
| Academic prompts by generator | Split across ChatGPT, DeepSeek, Gemini, and Doubao |
| Academic rewrites | Started in parallel |

The main insight was that a literature-only dataset would not be enough. The
teacher-test distribution required academic and likely poetry coverage as well.

## 3. Generator And Quality Notes

Expected generator roles:

| Generator | Role |
| --- | --- |
| ChatGPT | Conservative, human-like rewrites |
| DeepSeek | High-quality rewrite diversity |
| Gemini | Additional generator diversity, but with truncation risk |
| Doubao | Additional stylistic diversity, but sometimes expansion-heavy |

Quality checks emphasized:

1. Length ratio close to the source.
2. Lexical Jaccard not too high and not too low.
3. No copied source text.
4. No empty text.
5. No prompt leakage or meta commentary.
6. No silent truncation.
7. Preserved terminology for academic passages.

## 4. Immediate Next Steps At That Time

The recommended order for the next work block was:

1. Inspect all active generation outputs.
2. Add or improve truncation filtering scripts.
3. Produce a clean Gemini literature file.
4. Merge human seeds.
5. Build a literature-only full dataset as a pipeline sanity check.
6. Build a literature + academic dataset.
7. Train TF-IDF baselines on both datasets.
8. Fine-tune DeBERTa once the dataset was stable.

## 5. Scripts Planned Or Needed

Important scripts listed in the progress handoff:

```text
src/data/filter_truncated_rewrites.py
src/data/merge_human_seeds.py
src/data/build_full_dataset.py
src/data/inspect_human_seed.py
src/data/inspect_llm_rewrite.py
src/data/split_dataset_by_pair.py
src/models/train_tfidf_baseline.py
src/models/train_deberta.py
src/models/ensemble.py
```

The handoff also recommended making inspection scripts reusable across
literature, academic, and poetry data.

## 6. Training And Ablation Plan

Planned experiments:

| Experiment | Data setting | Model | Purpose |
| --- | --- | --- | --- |
| E1 | Literature-only | TF-IDF | Basic baseline |
| E2 | Literature-only | DeBERTa | Neural baseline |
| E3 | Literature + academic | TF-IDF | Check academic-domain value |
| E4 | Literature + academic | DeBERTa | Main multi-domain model |
| E5 | Literature + academic + poetry | DeBERTa | Optional final enhancement |
| E6 | Full data | TF-IDF + DeBERTa ensemble | Final submission model |

Generator ablations were also proposed:

| Ablation | Purpose |
| --- | --- |
| Remove Gemini | Check whether Gemini noise hurts |
| Remove Doubao | Check whether expansion-heavy rewrites hurt |
| DeepSeek + ChatGPT only | High-quality generator subset |
| All generators | Maximum diversity |

## 7. Risks Identified

| Risk | Mitigation |
| --- | --- |
| Gemini silent truncation | Use finish reasons, ending-completeness checks, filtering, and reruns |
| Academic PDF parsing noise | Inspect academic seed and preserve metadata for error analysis |
| Generator imbalance | Use generator breakdowns and optionally control generator counts |
| Domain imbalance | Add academic and poetry data; consider domain-aware sampling if needed |
| Teacher-test leakage | Never use teacher-test labels for training, tuning, threshold choice, or prompt selection |

## 8. Final Snapshot From This Handoff

The main conclusion on 2026-05-20 was that the project was entering the final
training-data formation stage. The next successful milestone would be a clean
multi-domain dataset and a first reportable TF-IDF baseline.

This progress note has since been superseded by:

```text
PROJECT_REPORT.md
docs/rounds/round1_optimization_work_log_2026-05-21.md
docs/rounds/round2_95_optimization_plan_2026-05-21.md
```
