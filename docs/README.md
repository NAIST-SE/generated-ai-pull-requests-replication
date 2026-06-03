# GitHub Pages (docs)

This folder contains static, client-side tools for manual PR-description analysis.

## Available Tools

1. `index.html` - Blind Coder
   - Purpose: Run a blind coding session where you classify descriptions as Human or AI.
   - Inputs: `detection_results.csv` (new session), or existing `blind_coding_results.json/.csv` (results viewer).
   - Output: Export coded outcomes as JSON or CSV.

2. `inspector.html` - PR Description Inspector
   - Purpose: Manually inspect sampled PR descriptions with filtering before/during review.
   - Filters: Origin (`Human`/`AI`) and Context (`Original`, `Repo Path`, `PR Title`, `File Diff`, `File Contents`, `Linked Issues`, `Without File Contents`).
   - Inputs: CSV with PR descriptions (supports fields such as `text`, `entry_type`, and context from `context_level`/`variation`/`prompt_variation`/`entry_key`).
   - Output: Optional reviewer notes export as `inspector_notes.csv`.

## Quick Links (After Pages Is Enabled)

- Main Blind Coder: `https://<username>.github.io/<repository>/`
- Inspector: `https://<username>.github.io/<repository>/inspector.html`

How to enable Pages for this repo:

1. Push these files to the repository's default branch (e.g., `main`).
2. In GitHub, open the repository Settings → Pages.
3. Under "Source", select the branch (e.g., `main`) and the folder `/docs`.
4. Save. GitHub will publish the site at `https://<username>.github.io/<repository>/` (or a project page URL).

Notes:

- All tools are entirely client-side and read CSV/JSON files you upload in the browser.
- Allow a few minutes for GitHub Pages to build the site after enabling it.
