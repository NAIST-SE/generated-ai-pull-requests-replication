#!/usr/bin/env python3
"""
Quick check whether PRs from the selected PRs CSV exist on GitHub.

Saves a CSV of missing/unreachable PRs to ../datasets/prs_missing_check.csv

Usage:
    pyenv exec python check_prs_exist.py [--input ../datasets/selected_prs_sampled.csv] [--output ../datasets/prs_missing_check.csv]
"""

from pathlib import Path
import argparse
import csv
import itertools
import random
import time
import requests
import sys


TOKENS_FILE = Path("./env/tokens.txt")
DEFAULT_INPUT = Path("../datasets/selected_prs_sampled.csv")
DEFAULT_OUTPUT = Path("../datasets/prs_missing_check.csv")
REQUEST_DELAY = 0.5


def load_tokens(tokens_file: Path):
    if not tokens_file.exists():
        return [], None
    with open(tokens_file, "r") as fh:
        tokens = [t.strip() for t in fh.read().splitlines() if t.strip()]
    if not tokens:
        return [], None
    start_index = random.randint(0, len(tokens) - 1)
    rotated = tokens[start_index:] + tokens[:start_index]
    return tokens, itertools.cycle(rotated)


def get_repo_from_row(row: dict):
    # Try same keys as fetch_pr_files.py
    for key in ("repository_name_with_owner", "search_repository", "repository"):
        if key in row and row.get(key) and str(row.get(key)).strip():
            return str(row.get(key)).strip()
    return None


def check_pr_exists(repo: str, pr_number: int, token_cycle, timeout=10):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {}
    if token_cycle is not None:
        token = next(token_cycle)
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        return resp.status_code, resp.text
    except requests.RequestException as e:
        return None, str(e)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--delay", type=float, default=REQUEST_DELAY)
    p.add_argument("--max", type=int, default=None, help="Max number of PRs to check")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 2

    tokens, token_cycle = load_tokens(TOKENS_FILE)
    if tokens:
        print(f"Loaded {len(tokens)} tokens (rotating)")
    else:
        print("No tokens found; requests will be unauthenticated and may hit rate limits")

    missing = []
    other_errors = []
    total = 0

    # Determine total rows for progress reporting
    try:
        with open(input_path, newline='', encoding='utf-8') as cntfh:
            total_rows = sum(1 for _ in cntfh) - 1  # subtract header
            if total_rows < 0:
                total_rows = 0
    except Exception:
        total_rows = None

    start_time = time.time()
    processed = 0

    # Read CSV in a streaming fashion and report progress/ETA
    import csv
    with open(input_path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if args.max and i >= args.max:
                break
            processed += 1
            total += 1
            repo = get_repo_from_row(row)
            try:
                pr_number = int(row.get('pull_number') or row.get('pr_number') or row.get('number'))
            except Exception:
                pr_number = None

            if not repo or not pr_number:
                other_errors.append({
                    'row_index': i,
                    'repo': repo,
                    'pull_number': row.get('pull_number'),
                    'note': 'Missing repo or pull_number'
                })
                continue

            # Print progress header for this check (periodic to avoid excess output)
            if total_rows:
                elapsed = time.time() - start_time
                avg = elapsed / processed if processed else 0
                remaining = max(0, total_rows - processed)
                eta = avg * remaining
                if processed == 1 or (processed % 10 == 0):
                    print(f"[{processed}/{total_rows}] Checking {repo}#{pr_number} — elapsed: {int(elapsed)}s, ETA: {int(eta)}s", flush=True)
            else:
                if processed == 1 or (processed % 10 == 0):
                    print(f"[{processed}] Checking {repo}#{pr_number}", flush=True)

            status, text = check_pr_exists(repo, pr_number, token_cycle)
            if status == 200:
                # exists
                pass
            elif status == 404:
                missing.append({'repo': repo, 'pull_number': pr_number, 'status': 404})
                print(f"[{i+1}] MISSING: {repo}#{pr_number}")
            elif status == 403:
                print(f"[{i+1}] RATE/LIMIT or Forbidden for {repo}#{pr_number} (403). Trying next token and retrying once...")
                # try once more with next token if available
                if token_cycle is not None:
                    status2, _ = check_pr_exists(repo, pr_number, token_cycle)
                    if status2 == 200:
                        continue
                    if status2 == 404:
                        missing.append({'repo': repo, 'pull_number': pr_number, 'status': 404})
                        print(f"[{i+1}] MISSING after retry: {repo}#{pr_number}")
                        continue
                other_errors.append({'repo': repo, 'pull_number': pr_number, 'status': status, 'text': text})
            elif status is None:
                other_errors.append({'repo': repo, 'pull_number': pr_number, 'status': None, 'text': text})
            else:
                # record other statuses (e.g., 301, 500, 422)
                other_errors.append({'repo': repo, 'pull_number': pr_number, 'status': status, 'text': text[:200] if isinstance(text, str) else text})

            time.sleep(args.delay)

    # Save missing to CSV
    if missing:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as outfh:
            w = csv.DictWriter(outfh, fieldnames=['repo', 'pull_number', 'status'])
            w.writeheader()
            for r in missing:
                w.writerow(r)
        print(f"Saved {len(missing)} missing PRs to {output_path}")
    else:
        print("No missing PRs found")

    # Print summary of other errors
    if other_errors:
        print("Other errors encountered:")
        for e in other_errors[:10]:
            print(" -", e)
        print(f"... and {len(other_errors)-10} more" if len(other_errors) > 10 else "")

    print(f"Checked {total} PRs")
    return 0


if __name__ == '__main__':
    sys.exit(main())
