#!/usr/bin/env python3
"""
PR Diffs & File Contents Extraction

This script fetches PR diffs and file contents using the GitHub API for the selected PRs.

Input: datasets/selected_prs.csv
Output: datasets/selected_prs_with_diffs_and_file_contents.csv

Usage:
    python fetch_pr_files.py [--skip-file-content]
"""

import argparse
import datetime
import pandas as pd
import requests
import pickle
import time
import os
import json
import itertools
import random
from pathlib import Path
import threading
import concurrent.futures

# === Configuration ===
TOKENS_FILE = "./env/tokens.txt"
REQUEST_DELAY = 1.0
DATASETS_DIR = "../datasets"
INPUT_FILE = f"{DATASETS_DIR}/selected_prs_sampled.csv"
OUTPUT_FILE = f"{DATASETS_DIR}/selected_prs_with_diffs_and_file_contents.csv"
PROGRESS_FILE = f"{DATASETS_DIR}/pr_files_progress.pkl"
LOG_FILE = f"{DATASETS_DIR}/pr_files_output.log"
MAX_WORKERS = 10

log_lock = threading.Lock()

def log_activity(activity: str, log_path=LOG_FILE):
    """Log activity with timestamp"""
    log = f"{datetime.datetime.now()}: {activity}\n"
    with log_lock:
        with open(log_path, "a") as log_file:
            log_file.write(log)
        print(activity)

token_lock = threading.Lock()
def get_next_token(token_cycle):
    with token_lock:
        return next(token_cycle)


def load_tokens(tokens_file):
    """Load and rotate GitHub API tokens"""
    with open(tokens_file, "r") as file:
        tokens = [token.strip() for token in file.read().splitlines() if token.strip()]
    if not tokens:
        raise ValueError("No tokens found in token file.")
    start_index = random.randint(0, len(tokens) - 1)
    rotated_tokens = tokens[start_index:] + tokens[:start_index]
    return tokens, itertools.cycle(rotated_tokens)


def get_file_size_and_content(contents_url, token_cycle, save_content=False, log_path=None):
    """Get file size and optionally decoded content from GitHub contents API"""
    token = get_next_token(token_cycle)
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(contents_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            size = data.get("size")
            content = None
            if save_content and "content" in data and data.get("encoding") == "base64":
                import base64
                try:
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                except Exception as e:
                    if log_path:
                        log_activity(f"Failed to decode content for {contents_url}: {e}", log_path)
                    content = None
            return size, content
        elif resp.status_code == 403 and "rate limit" in resp.text.lower():
            if log_path:
                log_activity("Rate limit hit during contents_url request, switching token...", log_path)
            return get_file_size_and_content(contents_url, token_cycle, save_content, log_path)
        else:
            if log_path:
                log_activity(f"Failed to get size ({resp.status_code}) for {contents_url}", log_path)
            return None, None
    except requests.RequestException as e:
        if log_path:
            log_activity(f"Request failed for {contents_url}: {e}", log_path)
        return None, None


def fetch_pr_files(repo, pr_number, token_cycle, log_path=None):
    """Fetch PR files with pagination"""
    all_files = []
    page = 1
    while True:
        token = get_next_token(token_cycle)
        headers = {"Authorization": f"Bearer {token}"}
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files?per_page=100&page={page}"

        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                all_files.extend(data)
                if len(data) < 100:
                    break  # last page
                else:
                    page += 1
                    time.sleep(REQUEST_DELAY)
            elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                if log_path:
                    log_activity("Rate limit hit, switching token...", log_path)
                continue
            else:
                if log_path:
                    log_activity(
                        f"Failed ({resp.status_code}) on {repo}#{pr_number}: {resp.json().get('message') if resp.text else 'Unknown error'}",
                        log_path
                    )
                break
        except requests.RequestException as e:
            if log_path:
                log_activity(f"Request failed for {repo}#{pr_number}: {e}", log_path)
            break

    return all_files


def process_selected_prs(save_file_content=True):
    """Process the selected PRs and fetch their files"""
    log_activity("🚀 Starting PR files extraction...")
    
    # Load tokens
    try:
        all_tokens, token_cycle = load_tokens(TOKENS_FILE)
        log_activity(f"Loaded {len(all_tokens)} GitHub API tokens")
    except Exception as e:
        log_activity(f"Error loading tokens: {e}")
        return False

    # Load selected PRs CSV
    try:
        df = pd.read_csv(INPUT_FILE)
        log_activity(f"Loaded {len(df)} selected PRs from {INPUT_FILE}")
    except FileNotFoundError:
        log_activity(f"Error: {INPUT_FILE} not found.")
        return False

    # Load previous progress
    progress_lock = threading.Lock()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "rb") as f:
            progress = pickle.load(f)
        log_activity(f"Loaded progress for {len(progress)} PRs.")
    else:
        progress = {}

    def process_row(i, row):
        # Extract repository info - check multiple possible column names
        if "repository_name_with_owner" in row and pd.notna(row["repository_name_with_owner"]):
            repository = row["repository_name_with_owner"]
        elif "search_repository" in row and pd.notna(row["search_repository"]):
            repository = row["search_repository"]
        else:
            log_activity(f"[{i + 1}/{len(df)}] No repository column found, skipping row")
            return
            
        pull_number = row["pull_number"]
        pr_id = row["id"]
        
        # Create unique key for this PR
        key = f"{repository}#{pull_number}"
        
        with progress_lock:
            if key in progress:
                log_activity(f"[{i + 1}/{len(df)}] Skipping already processed {key}")
                return

        log_activity(f"[{i + 1}/{len(df)}] Fetching {key} (PR ID: {pr_id})...")
        pr_files = fetch_pr_files(repository, pull_number, token_cycle, LOG_FILE)

        if pr_files:
            pr_total_size = 0
            for file_data in pr_files:
                contents_url = file_data.get("contents_url")
                if contents_url:
                    size, content = get_file_size_and_content(contents_url, token_cycle, save_file_content, LOG_FILE)
                    file_data["file_size_bytes"] = size
                    if save_file_content:
                        file_data["file_content"] = content
                    if size is not None:
                        pr_total_size += size
                    time.sleep(REQUEST_DELAY)

            with progress_lock:
                # Store all the row data plus the fetched files
                progress[key] = {
                    **row.to_dict(),  # Include all original columns
                    "files": pr_files,
                    "pr_total_size_bytes": pr_total_size,
                }

                # Save progress
                with open(PROGRESS_FILE, "wb") as f:
                    pickle.dump(progress, f)

            log_activity(
                f"✅ [{i + 1}/{len(df)}] Saved progress for {key} ({len(pr_files)} files, {pr_total_size} bytes total)"
            )
        else:
            log_activity(f"⚠️ [{i + 1}/{len(df)}] No files fetched for {key}")

        time.sleep(REQUEST_DELAY)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_row, i, row) for i, (_, row) in enumerate(df.iterrows())]
        concurrent.futures.wait(futures)

    log_activity(f"✅ Done. Fetched data for {len(progress)} PRs.")

    # Convert to CSV format
    csv_rows = []
    for key, pr_data in progress.items():
        files = pr_data.get("files", [])
        pr_total_size_bytes = pr_data.get("pr_total_size_bytes", None)

        if not files:
            # Even if no files, include the PR with null file data
            csv_rows.append({
                **{k: v for k, v in pr_data.items() if k not in ["files", "pr_total_size_bytes"]},
                "filename": None,
                "status": None,
                "additions": None,
                "deletions": None,
                "changes": None,
                "sha": None,
                "blob_url": None,
                "raw_url": None,
                "patch": None,
                "file_size_bytes": None,
                "file_content": None if save_file_content else None,
                "pr_total_size_bytes": pr_total_size_bytes,
            })
        else:
            # One row per file
            for f in files:
                csv_rows.append({
                    **{k: v for k, v in pr_data.items() if k not in ["files", "pr_total_size_bytes"]},
                    "filename": f.get("filename"),
                    "status": f.get("status"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                    "changes": f.get("changes"),
                    "sha": f.get("sha"),
                    "blob_url": f.get("blob_url"),
                    "raw_url": f.get("raw_url"),
                    "patch": f.get("patch", None),
                    "file_size_bytes": f.get("file_size_bytes"),
                    "file_content": f.get("file_content") if save_file_content else None,
                    "pr_total_size_bytes": pr_total_size_bytes,
                })

    # Save to CSV
    pd.DataFrame(csv_rows).to_csv(OUTPUT_FILE, index=False)
    log_activity(f"📄 Saved CSV results to {OUTPUT_FILE}")
    log_activity(f"🎉 Processing completed successfully!")
    
    return True


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Extract PR diffs and file contents")
    parser.add_argument(
        "--skip-file-content", 
        action="store_true", 
        help="Skip fetching actual file contents (faster, smaller output)"
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(DATASETS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(TOKENS_FILE), exist_ok=True)
    
    # Process PRs
    success = process_selected_prs(save_file_content=not args.skip_file_content)
    
    if success:
        log_activity("✅ All PRs processed successfully!")
        return 0
    else:
        log_activity("❌ Processing failed!")
        return 1


if __name__ == "__main__":
    exit(main())
