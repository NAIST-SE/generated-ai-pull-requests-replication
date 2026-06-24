"""
Assembles individual figure PDFs into textual_features_boxplots.pdf
in the correct page order for the paper.

Page order:
  1  – verbosity_word_count.pdf      (main text, fig:verbosity_boxplots)
  2  – fkgl.pdf                      (appendix, fig:fkgl_boxplots)
  3  – ttr.pdf                       (appendix, fig:ttr_boxplots)
  4  – mtld_bidirectional.pdf        (appendix, fig:mtld_boxplots)
  5  – lexical_density.pdf           (appendix, fig:ld_boxplots)
  6  – rouge.pdf                     (main text, fig:rouge_boxplots)
  7  – bertscore.pdf                 (main text, fig:bertscore_boxplots)
  8  – bm25.pdf                      (main text, fig:bm25_boxplots)
  9  – codebert.pdf                  (main text, fig:codebert_boxplots)
  10 – graphcodebert.pdf             (main text, fig:graphcodebert_boxplots)

Run from anywhere:
  python assemble_textual_features_pdf.py

Requires: pypdf  (pip install pypdf)
"""

from pathlib import Path
from pypdf import PdfWriter

FIGURES_DIR = Path(__file__).parent / "outputs" / "figures_pdf"
OUTPUT_PATH = FIGURES_DIR / "textual_features_boxplots.pdf"

PAGE_ORDER = [
    "verbosity_word_count.pdf",
    "fkgl.pdf",
    "ttr.pdf",
    "MTLD_Bidirectional_Distribution_Human_vs_Contexts.pdf",
    "lexical_density.pdf",
    "rouge.pdf",
    "bertscore.pdf",
    "bm25.pdf",
    "codebert.pdf",
    "graphcodebert.pdf",
]

def assemble():
    writer = PdfWriter()
    missing = []

    for fname in PAGE_ORDER:
        path = FIGURES_DIR / fname
        if not path.exists():
            missing.append(fname)
            print(f"  MISSING: {fname}")
            continue
        writer.append(str(path))
        print(f"  Added:   {fname}")

    if missing:
        print(f"\n{len(missing)} file(s) missing — run the notebooks first to generate them.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        writer.write(f)
    print(f"\nAssembled {len(PAGE_ORDER)} pages → {OUTPUT_PATH}")

if __name__ == "__main__":
    assemble()
