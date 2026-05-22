# Release Quality Audit

Updated: 2026-05-22

This audit records the final local deliverable review and the GitHub-facing
repository surface before publishing.

## Report QA

Local deliverable:

```text
report/final_report.pdf
report/final_report.tex
```

Checks performed:

| Check | Result |
| --- | --- |
| Rendered PDF page count | 13 pages |
| Page geometry | A4, portrait |
| Visual contact sheet | `report/qa_review/report_contact_sheet.png` |
| Text extraction | Chinese and English text extracted correctly with `pdftotext -enc UTF-8` |
| LaTeX log scan | No fatal errors, no overfull boxes, no missing-character errors |
| Non-blocking log notes | CJK/font redefinition warnings and two mild underfull boxes |

Visual review found no obvious overlap, clipping, chart overflow, missing glyphs,
or broken table layout. The cover, abstract, table of contents, body sections,
figures, captions, and conclusion pages are consistent enough for the final
course report and local archival copy.

## Presentation QA

Local deliverable:

```text
presentation/llm_text_detector_final_presentation.pptx
```

Checks performed:

| Check | Result |
| --- | --- |
| Slide count | 12 slides |
| Export resolution | 1920 x 1080 PNG per slide |
| Visual contact sheet | `presentation/qa_review/ppt_contact_sheet.png` |
| Exported slide previews | `presentation/qa_review/slide_01.png` through `slide_12.png` |
| Package text extraction | Slide titles and body text present for all slides |
| Media inventory | 4 embedded media assets, approximately 697 KB total |

Visual review found a coherent system, readable charts, stable footers, and no
visible overlap or off-slide text. The deck is concise and presentation-ready.
The internal OOXML metadata still reflects the exporter rather than a publisher
toolchain; because the PPTX is a local-only course deliverable, this is not a
blocking repository issue.

## Filename Audit

Remote audit source:

```powershell
git ls-tree -r --name-only origin/main
```

Result:

| Surface | Result |
| --- | --- |
| Remote tracked files before this release patch | 349 files |
| Release-candidate tracked files after this patch | 351 files |
| Whitespace in tracked paths | none found |
| Non-ASCII tracked paths | none found |
| Case-collision tracked paths | none found |
| Temporary tracked filenames | none found |
| Report/PPT binaries on remote | none tracked |

Local-only QA images exported by PowerPoint originally used localized slide
names. They were normalized to ASCII `slide_XX.png` filenames under
`presentation/qa_review/`.

## Publication-Ready Repository Surface

The GitHub-facing repository is intentionally split from local course
deliverables:

| Surface | Publication decision |
| --- | --- |
| `README.md` | public overview and setup |
| `PROJECT_REPORT.md` | public consolidated result narrative |
| `CITATION.cff` | public citation metadata |
| `requirements.txt` | public Python environment |
| `src/` | public data/model/evaluation code |
| `docs/final/` | public model card, manifest, and release audit |
| `docs/rounds/` | public optimization history and promotion decisions |
| `data/raw/teacher_test.json` | tracked diagnostic input supplied for this project |
| `data/processed/` | tracked processed reproducibility datasets and reports |
| `outputs/models/` | final baseline artifacts only; large neural weights via Git LFS |
| `outputs/figures/publication/` | final publication figures |
| `report/` | local-only report source, build products, PDF, and QA previews |
| `presentation/` | local-only PPTX and QA previews |
| `.env`, `.venv/`, IDE files, caches | local-only |

The final public story is conservative: Step7 DeBERTa plus TF-IDF remains the
submitted system, while later residual-repair rounds are documented as
diagnostic evidence rather than promoted models.

## Pre-Push Verification

Recommended final commands:

```powershell
git fetch origin
git status --short --branch
git check-ignore -v .env report/final_report.pdf presentation/llm_text_detector_final_presentation.pptx
git check-attr -a -- outputs/models/deberta_lit_academic_poetry_step7_combined/best_model/model.safetensors outputs/models/tfidf_lit_academic_poetry/word_tfidf_vectorizer.pkl
.\.venv\Scripts\python.exe -m compileall src
git diff --cached --check
git push origin main
```
