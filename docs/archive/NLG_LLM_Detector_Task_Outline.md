# NLG Course Project Task Outline

Archived: 2026-05-21

This archived document is the English version of the original task outline for
the LLM Text Detector project. It has been consolidated into the public project
surface so that all Markdown documentation uses English consistently.

## Project Title

Detecting LLM-Rewritten Text with DeBERTa and Multi-level TF-IDF Features

## 1. Background

Large language models can generate fluent, grammatically correct, and
semantically coherent text. Traditional text classification or plagiarism
detection methods are not enough to determine whether an English passage was
written by a human or generated, rewritten, modernized, or polished by an LLM.

This course project focuses on a binary classification task:

| Label | Meaning |
| --- | --- |
| `0` | Human-written text |
| `1` | LLM-generated or LLM-rewritten text |

The instructor-provided test distribution is not a generic AI essay detection
setting. It resembles LLM rewrite detection for literature and academic text.
The samples include literary prose, archaic English, poetry, narrative fiction,
NLP and computational linguistics paragraphs, and corresponding LLM rewrites or
imitations.

## 2. Task Definition

Input:

```json
{
  "text": "The chamber within was illuminated solely by the amber radiance..."
}
```

Output:

```json
{
  "label": 1,
  "probability": 0.83
}
```

The model should detect signals such as:

1. Modernization of literary style.
2. Smoothing of archaic English, dialect, poetry, or author-specific style.
3. Academic paragraphs becoming more explanatory, templated, or fluent.
4. Vocabulary replaced by more modern, abstract, or polished expressions.
5. Syntax becoming overly regular.
6. Loss of natural irregularity from the original human text.

## 3. Overall System Design

The planned detector is a hybrid system:

```text
Input text
   |
   |-- Branch A: DeBERTa-v3-base classifier
   |      -> P_deberta(label=1)
   |
   |-- Branch B: Word/char TF-IDF + Logistic Regression
          -> P_tfidf(label=1)

Final:
P_final = alpha * P_deberta + (1 - alpha) * P_tfidf
```

The DeBERTa branch captures semantic, discourse, and deeper style patterns. The
TF-IDF branch captures lexical replacement, punctuation, spelling, character
n-grams, and surface style.

## 4. Data Plan

The original outline called for three complementary human-data domains:

| Domain | Source idea | Purpose |
| --- | --- | --- |
| Literature | Project Gutenberg fiction | Main literary rewrite detection |
| Academic | ACL-OCL / computational linguistics paragraphs | Match academic teacher-test samples |
| Poetry / archaic English | Public-domain poetry and old-style text | Cover line breaks, rhyme, and unusual syntax |

Each human passage should receive one or more LLM rewrites. The final dataset
must keep each human source and its rewrites under the same `pair_id` so that
pair-safe splitting can prevent source leakage across train, validation, and
internal-test splits.

## 5. Generator Strategy

The planned rewrite generators were:

| Generator | Intended role |
| --- | --- |
| ChatGPT | Human-like conservative rewrites and polished paraphrases |
| DeepSeek | High-quality rewrite diversity |
| Gemini | Additional generator diversity |
| Doubao | Additional stylistic diversity |

Prompt styles should cover modernization, conservative paraphrase, old-fiction
style, academic naturalization, and poetry-preserving rewrites. Quality filters
should reject empty text, prompt leakage, excessive repetition, copied text,
truncation, and extreme length ratios.

## 6. Modeling Plan

The task outline proposed this implementation sequence:

1. Build human seed data.
2. Prepare rewrite prompts with generator assignments.
3. Generate LLM rewrites.
4. Filter and inspect rewrite quality.
5. Build the full labeled dataset.
6. Split by `pair_id`.
7. Train TF-IDF Logistic Regression as the fast baseline.
8. Fine-tune DeBERTa-v3-base.
9. Tune the DeBERTa + TF-IDF probability ensemble on validation.
10. Evaluate on internal test and only then run the teacher test.

## 7. Evaluation Plan

Core metrics:

```text
accuracy
precision
recall
F1
ROC-AUC
confusion matrix
```

Breakdowns should include:

```text
domain: literature / academic / poetry
generator: ChatGPT / DeepSeek / Gemini / Doubao
error type: false positive / false negative
```

The teacher test should not be used for training, prompt selection, threshold
tuning, or model selection.

## 8. Report Claims Intended By The Outline

Supported claims:

1. A hybrid neural + lexical detector is appropriate for LLM rewrite detection.
2. TF-IDF is not just a weak baseline; it captures useful surface artifacts.
3. DeBERTa adds semantic and discourse-level sensitivity.
4. Pair-safe splitting is necessary to reduce source-passage leakage.
5. Domain and generator breakdowns are more informative than a single accuracy
   number.

Claims to avoid:

1. The system is not a universal AI-text detector.
2. Teacher-test performance should not be generalized to arbitrary domains or
   future LLMs without additional evaluation.
3. Teacher-test labels must not be used as a tuning set unless explicitly
   labeled as post-hoc repair.

## 9. Current Replacement Documents

This archived outline has been superseded by:

```text
README.md
PROJECT_REPORT.md
docs/SECOND_ROUND_95_OPTIMIZATION_PLAN.md
```

Those files contain the current public-facing project summary, final results,
and second-round optimization plan.
