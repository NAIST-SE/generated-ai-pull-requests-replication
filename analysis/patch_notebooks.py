"""
Patch all RQ3 plotting notebooks to:
  1. Remove fig.suptitle / ax.set_title / plt.title calls
  2. Add fontsize=12 to xlabel/ylabel where missing
  3. Add labelsize=10 to tick_params
  4. Fix tight_layout (remove the 0.97 rect that compensated for suptitle)
  5. Add plt.savefig(...) before plt.show() in main context-comparison cells

Run from:  phase3/analysis/
"""

import json, re
from pathlib import Path

OUT_DIR = Path(__file__).parent / "outputs" / "figures_pdf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Low-level line transformations
# ---------------------------------------------------------------------------

TITLE_PAT = re.compile(
    r'^\s*(fig\.suptitle|ax\.set_title|plt\.title|axes\[\d+\]\.set_title)\s*\('
)

def transform_line(line: str):
    """Return None to drop the line, else return (possibly modified) line."""

    # 1. Drop title lines
    if TITLE_PAT.search(line):
        return None

    # 2. Add fontsize=12 to xlabel / ylabel if missing
    for fn in ('set_xlabel', 'set_ylabel', 'plt.xlabel', 'plt.ylabel'):
        if fn in line and 'fontsize' not in line:
            line = re.sub(
                rf'({re.escape(fn)}\s*\()([^)]*)\)',
                lambda m: f"{m.group(1)}{m.group(2).rstrip()}, fontsize=12)",
                line,
            )
            break

    # 3. Add labelsize=10 to tick_params if missing
    if 'tick_params' in line and 'labelsize' not in line:
        line = re.sub(
            r'(tick_params\s*\()([^)]*)\)',
            lambda m: f"{m.group(1)}{m.group(2).rstrip()}, labelsize=10)",
            line,
        )

    # 4. Fix tight_layout rect (remove the suptitle compensation)
    if 'tight_layout' in line and ('rect' in line):
        line = re.sub(r'\(rect\s*=\s*[\[\(][^\]\)]+[\]\)]\)', '()', line)

    return line


def patch_source(source_lines, savefig_path=None):
    """
    Transform a cell's source lines.
    If savefig_path is given, insert plt.savefig(...) before plt.show().
    """
    result = []
    for line in source_lines:
        transformed = transform_line(line)
        if transformed is None:
            continue
        # Insert savefig before plt.show() if this is a main cell
        if savefig_path and 'plt.show()' in transformed:
            indent = len(line) - len(line.lstrip())
            result.append(' ' * indent + f"plt.savefig(r'{savefig_path}', bbox_inches='tight', dpi=150)\n")
        result.append(transformed)
    return result


# ---------------------------------------------------------------------------
# Per-notebook patches
# ---------------------------------------------------------------------------

def patch_notebook(nb_path: Path, patches: dict) -> None:
    """
    patches: {cell_index: savefig_path_or_None}
      - cell_index: which cell to patch
      - savefig_path: path string for savefig (None = patch but don't save)
    """
    with open(nb_path) as f:
        nb = json.load(f)

    for cell_idx, savefig_path in patches.items():
        cell = nb['cells'][cell_idx]
        cell['source'] = patch_source(cell['source'], savefig_path)

    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Patched: {nb_path.name}")


base = Path(__file__).parent

# Map: notebook filename -> {cell_index: output_pdf_filename or None}
# None means patch for style but no savefig (exploratory cells)
NOTEBOOK_PATCHES = {
    'analysis_rq3_word_sentence_count.ipynb': {
        13: None,   # exploratory: Human vs AI
        15: None,   # exploratory: by Model
        17: str(OUT_DIR / 'verbosity_word_count.pdf'),
        19: str(OUT_DIR / 'verbosity_sentence_count.pdf'),
    },
    'analysis_rq3_reading_ease.ipynb': {
        13: None,   # exploratory
        15: str(OUT_DIR / 'fkgl.pdf'),
    },
    'analysis_rq3_ttr.ipynb': {
        32: None,   # exploratory
        33: None,   # exploratory
        37: str(OUT_DIR / 'ttr.pdf'),
    },
    'analysis_rq3_lexical_density.ipynb': {
        12: str(OUT_DIR / 'lexical_density.pdf'),  # all figures in one cell; savefig fires on last plt.show
    },
    'analysis_rq3_rouge.ipynb': {
        15: str(OUT_DIR / 'rouge.pdf'),
    },
    'analysis_rq3_bertscore.ipynb': {
        18: str(OUT_DIR / 'bertscore.pdf'),
    },
    'analysis_rq3_bm25.ipynb': {
        17: str(OUT_DIR / 'bm25.pdf'),
    },
    'analysis_rq3_codebert.ipynb': {
        15: str(OUT_DIR / 'codebert.pdf'),
    },
    'analysis_rq3_graphcodebert.ipynb': {
        15: str(OUT_DIR / 'graphcodebert.pdf'),
    },
}

# MTLD is handled separately because it uses a helper function
def patch_mtld(nb_path: Path) -> None:
    with open(nb_path) as f:
        nb = json.load(f)

    # Patch exploratory cells (33, 34)
    for idx in [33, 34]:
        nb['cells'][idx]['source'] = patch_source(nb['cells'][idx]['source'], None)

    # Patch the main function cell (38) — inject savefig via output_path param
    cell = nb['cells'][38]
    new_src = []
    for line in cell['source']:
        t = transform_line(line)
        if t is None:
            continue
        # Inject savefig inside the function, before plt.show()
        if 'plt.show()' in t:
            indent = len(line) - len(line.lstrip())
            new_src.append(' ' * indent + f"plt.savefig(str(OUT_DIR / (figure_title.replace(' ', '_').replace(':', '') + '.pdf')), bbox_inches='tight', dpi=150)\n")
        new_src.append(t)

    # Prepend OUT_DIR definition so the function can use it
    out_dir_line = f"OUT_DIR = Path(r'{str(OUT_DIR)}')\n"
    if out_dir_line not in ''.join(new_src):
        new_src.insert(0, "from pathlib import Path\n")
        new_src.insert(1, out_dir_line)

    cell['source'] = new_src
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"Patched: {nb_path.name}")


for nb_file, patches in NOTEBOOK_PATCHES.items():
    patch_notebook(base / nb_file, patches)

patch_mtld(base / 'analysis_rq3_mtld.ipynb')

print("\nAll notebooks patched.")
print(f"Individual PDFs will be saved to: {OUT_DIR}")
