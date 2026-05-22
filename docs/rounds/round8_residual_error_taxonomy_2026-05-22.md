# Round8-OneShot Residual Error Taxonomy

Updated: 2026-05-22

This document defines the residual buckets and data rules for the final
Round8-OneShot optimization pass. It follows
`docs/rounds/round8_one_shot_95_optimization_plan_2026-05-22.md`.

Current strict baseline:

```text
System: Step7 DeBERTa + TF-IDF ensemble
Teacher-test diagnostic: 274 / 300
95% target: at least 285 / 300
Needed net gain: +11 correct examples
```

## 1. Non-Negotiable Boundaries

Teacher-test is diagnostic-only.

Forbidden:

```text
1. Do not train on data/raw/teacher_test.json.
2. Do not tune thresholds, selectors, routers, guards, alphas, or rules on teacher-test.
3. Do not use teacher-test row ids, labels, text, or per-row errors to choose a candidate.
4. Do not generate rewrites from teacher-test text.
5. Do not promote a candidate because it looks good only on teacher-test.
```

Allowed:

```text
1. Use teacher-test text only for exact-text duplicate exclusion.
2. Run one frozen teacher-test diagnostic only after the non-teacher gate passes.
3. Report teacher-test results after the frozen diagnostic without post-hoc tuning.
```

The promotion gate must use non-teacher evidence:

```text
internal_test
residual_dev
residual_probe
hard human negatives
hard LLM positives
domain and generator breakdowns
leakage reports
```

## 2. Residual Problem Definition

The remaining Step7 errors are not one simple threshold problem.

Human false positives tend to be:

```text
polished, literary, archaic, poetic, formal, structured, or unusually clean
```

LLM false negatives tend to be:

```text
conservative rewrites that preserve style, terminology, sentence shape, or
line-level feel
```

Round8 therefore rebuilds the residual distribution first, then trains and gates
new signals on the ambiguous zone.

## 3. Human Hard-Negative Buckets

Human hard negatives are label `0`. They are expected to look deceptively LLM-like
to Step7, so they protect precision when new hard-positive signal is added.

| Bucket | Domain | Why Hard | Expected Step7 Error | Data Sources | Split Policy |
| --- | --- | --- | --- | --- | --- |
| `human_free_verse` | poetry | Short lineated text with clean imagery and modern diction | FP | Project Gutenberg poetry, unused poetry seed, poetry expansion seed | group by poem/book/source id |
| `human_classical_poetry` | poetry | Archaic diction, meter, compact syntax, high line-break density | FP | Gutenberg poetry, classical/archaic poem seeds | group by poem/book/source id |
| `human_archaic_prose` | literature | Old-fiction diction and unusual syntax can resemble generated style | FP | Gutenberg old prose, round4 old-prose mirrors | group by book/source id |
| `human_polished_literary_prose` | literature | Smooth, ornate, well-edited prose overlaps with LLM fluency cues | FP | Gutenberg fiction, round3/round4 hard human mirrors | group by book/source id |
| `human_formal_academic` | academic | Dense technical language, citations, and structured claims | FP | ACL-OCL paragraphs, academic hard negatives | group by paper id |
| `human_structured_explanatory` | mixed | Clean explanatory paragraphs with explicit discourse markers | FP | academic/literature human sources | group by source doc id |
| `human_literary_short_fragment` | literature | Short polished fragments give little evidence for human authorship | FP | short Gutenberg fragments, round4/round7 mined mirrors | group by source doc id |
| `human_old_fiction_dialogue` | literature | Dialogue, quotes, archaic phrasing, and narration shifts | FP | public-domain fiction dialogue passages | group by book/source id |

## 4. LLM Hard-Positive Buckets

LLM hard positives are label `1`. They are expected to look deceptively human-like
to Step7, so they target residual false negatives.

| Bucket | Domain | Why Hard | Expected Step7 Error | Data Sources | Split Policy |
| --- | --- | --- | --- | --- | --- |
| `llm_conservative_chatgpt_paraphrase` | mixed | ChatGPT rewrites preserve meaning and sound naturally edited | FN | existing ChatGPT hard-positive rewrites, future conservative rewrites | group by original human source |
| `llm_old_fiction_style_rewrite` | literature | LLM keeps old prose rhythm and diction close to the source | FN | old-prose rewrite pools, round4/round7 mined safe examples | group by source passage |
| `llm_archaic_poetry_rewrite` | poetry | LLM keeps imagery, line breaks, and archaic word choice | FN | poetry rewrite pools, poetry hard positives | group by poem/source id |
| `llm_academic_term_preserving_paraphrase` | academic | Technical terms, citations, and claims stay mostly unchanged | FN | academic rewrite pools, round2/round4 academic hard positives | group by paper id |
| `llm_low_temperature_minimal_rewrite` | mixed | Surface wording changes are small and lexical overlap is high | FN | quality metadata with high lexical Jaccard, conservative rewrite prompts | group by original pair id |
| `llm_style_preserving_dialogue_rewrite` | literature | Dialogue shape and narration style survive the rewrite | FN | dialogue rewrite pools or future train-only generation | group by source passage |
| `llm_high_jaccard_rewrite` | mixed | Rewrite is close to source but still machine-produced | FN | quality metadata with high but accepted lexical overlap | group by original pair id |
| `llm_human_like_formal_paraphrase` | academic/general | Formal, clean paraphrase lacks obvious LLM artifacts | FN | academic/general hard-positive rewrites | group by original source |

## 5. Required Metadata

Every residual candidate should carry enough metadata for leakage checks and
breakdown reporting.

Required fields:

```text
id
text
label
round8_bucket
round8_bucket_family
round8_source_stage
domain
generator
origin
source_file
source_doc_id
source_pair_id
pair_id
split_group
quality_flags
```

Recommended model/gate fields after Step7 scoring:

```text
p_step7
p_deberta_step7
p_tfidf
step7_pred
step7_correct
ambiguous_zone
```

## 6. Candidate Pool Policy

The candidate pool is not the training set. It is the broad non-teacher search
surface from which hard residuals are selected after Step7 scoring.

Initial target:

```text
5000 to 10000 candidates
human hard negatives: roughly 45% to 55%
LLM hard positives: roughly 45% to 55%
```

Quality filters:

```text
1. Keep 20 to 300 word passages by default.
2. Allow poetry shorter than 20 words only if lineation is meaningful.
3. Remove empty, prompt-leaked, truncated, or malformed LLM outputs.
4. Remove exact text duplicates.
5. Remove exact teacher-test duplicates.
```

## 7. Hard Residual Selection Policy

After Step7 scoring, select genuinely hard samples.

Human hard negatives:

```text
label = 0
primary hard condition: p_step7 >= 0.55
very hard condition: p_step7 >= 0.65
```

LLM hard positives:

```text
label = 1
primary hard condition: p_step7 <= 0.45
very hard condition: p_step7 <= 0.35
```

Ambiguous-zone samples:

```text
0.35 <= p_step7 <= 0.65
```

Ambiguous-zone samples are reserved for selector/fusion training. High-confidence
Step7 samples should not be globally overwritten.

## 8. Split Policy

Residual train/dev/probe must be split by group, not by row.

Use the strongest available group key:

```text
source_doc_id
pair_id
source_pair_id
original_text_id
book_id
paper_id
poem_id
```

Required leakage checks:

```text
train/dev/probe group overlap = 0
train/dev/probe text overlap = 0
teacher-test exact duplicate = 0
held-out probe is not used for threshold or rule tuning
```

## 9. Promotion Gate Implications

Round8-OneShot can advance to frozen teacher-test only if the non-teacher gate
shows a large enough improvement to plausibly repair the 11-example gap.

Minimum gate expectations:

```text
internal_test F1 >= Step7 internal_test F1 - 0.003
residual_dev F1 >= Step7 residual_dev F1 + 0.04
residual_probe F1 >= Step7 residual_probe F1 + 0.03
human hard-negative induced FP <= 1
internal_test induced FP <= 1
hard-positive FN reduced by at least 8% relative
ChatGPT recall improves clearly
poetry recall does not drop by more than 2%
academic F1 does not drop by more than 1%
no leakage
```

If this gate does not pass, keep Step7 as final and do not run another
teacher-test diagnostic.
