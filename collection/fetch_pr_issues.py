#!/usr/bin/env python3
"""
PR Related Issues Extraction

This script fetches GitHub issues related to PRs from the selected_prs.csv dataset.

Input: datasets/selected_prs.csv
Output: datasets/selected_prs_with_related_issues.csv
"""

import datetime
import pandas as pd
import requests
import pickle
import time
import os
import sys
import json
import itertools
import random
import re
import threading
import concurrent.futures

# === Configuration ===
INPUT_CSV = "../datasets/selected_prs_sampled.csv"
OUTPUT_CSV = "../datasets/selected_prs_with_related_issues.csv"
OUTPUT_JSON = "../datasets/selected_prs_with_related_issues.json"
PICKLE_FILE = "../datasets/pr_issues_progress.pkl"
LOG_PATH = "../datasets/pr_issues_output.log"
TOKENS_FILE = "./env/tokens.txt"
REQUEST_DELAY = 1.0
DEFAULT_REPOSITORY = "JabRef/jabref"  # Fallback repository
MAX_WORKERS = 10

log_lock = threading.Lock()

def log_activity(activity: str, log_path=LOG_PATH):
    """Log activity with timestamp to specified log file"""
    log = f"{datetime.datetime.now()}: {activity}\n"
    with log_lock:
        # Ensure directory exists for log file
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(log)
        print(activity)


def load_tokens(tokens_file):
    """Load GitHub tokens from file and create a rotating iterator"""
    if not os.path.exists(tokens_file):
        raise FileNotFoundError(f"Tokens file not found: {tokens_file}")
    
    with open(tokens_file, "r") as file:
        tokens = [line.strip() for line in file.readlines() if line.strip()]
    
    if not tokens:
        raise ValueError("No tokens found in token file.")
    
    # Start from a random position to distribute load
    start_index = random.randint(0, len(tokens) - 1)
    rotated_tokens = tokens[start_index:] + tokens[:start_index]
    return tokens, itertools.cycle(rotated_tokens)


def parse_issue_numbers(issue_references_str):
    """
    Parse issue numbers from the issue_references column.
    Format: comma-separated issue numbers like "2890" or "3920,5131"
    """
    if pd.isna(issue_references_str) or not issue_references_str:
        return []

    try:
        # Split by comma and extract numbers
        issue_parts = str(issue_references_str).split(',')
        parsed_issues = []

        for issue_part in issue_parts:
            # Extract number, handling potential whitespace
            issue_str = issue_part.strip()
            if issue_str.isdigit():
                issue_num = int(issue_str)
                parsed_issues.append(issue_num)

        return parsed_issues
    except (ValueError, AttributeError) as e:
        log_activity(f"Error parsing issue numbers from '{issue_references_str}': {e}")
        return []

token_lock = threading.Lock()
def get_next_token(token_cycle):
    with token_lock:
        return next(token_cycle)

def fetch_issue(repo, issue_number, token_cycle, log_path):
    """Fetch issue data from GitHub API with error handling and token rotation"""
    token = get_next_token(token_cycle)
    headers = {"Authorization": f"Bearer {token}"}
    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403 and "rate limit" in resp.text.lower():
            log_activity(f"Rate limit hit while fetching issue {issue_number}, switching token...", log_path)
            return fetch_issue(repo, issue_number, token_cycle, log_path)  # retry with next token
        elif resp.status_code == 404:
            log_activity(f"Issue #{issue_number} not found in {repo}", log_path)
            return None
        else:
            log_activity(
                f"Failed to fetch issue #{issue_number} from {repo} ({resp.status_code}): {resp.json().get('message', 'Unknown error')}", 
                log_path
            )
            return None
    except requests.RequestException as e:
        log_activity(f"Request failed for {repo}#{issue_number}: {e}", log_path)
        return None


def fetch_issue_comments(comments_url, token_cycle, log_path):
    """Fetch all comments for an issue with pagination"""
    all_comments = []
    page = 1

    while True:
        token = get_next_token(token_cycle)
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{comments_url}?per_page=100&page={page}"

        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                all_comments.extend(data)
                if len(data) < 100:
                    break  # last page
                else:
                    page += 1
                    time.sleep(REQUEST_DELAY)
            elif resp.status_code == 403 and "rate limit" in resp.text.lower():
                log_activity(
                    f"Rate limit hit while fetching comments from {comments_url}, switching token...", 
                    log_path
                )
                continue  # retry with next token
            else:
                log_activity(
                    f"Failed to fetch comments ({resp.status_code}) from {comments_url}", 
                    log_path
                )
                break
        except requests.RequestException as e:
            log_activity(f"Request failed for comments {comments_url}: {e}", log_path)
            break

    return all_comments


def format_comments(comments, issue_number):
    """Format comments into a single string with issue context"""
    if not comments:
        return ""

    formatted_comments = []
    for i, comment in enumerate(comments, 1):
        author = comment.get("user", {}).get("login", "Unknown")
        body = comment.get("body", "").strip()
        formatted_comments.append(f"Comment #{i} by {author} in issue #{issue_number}: {body}")

    return "\n\n".join(formatted_comments)


def clean_for_json(obj):
    """Recursively clean data to remove NaN values and make it JSON serializable"""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Timestamp, pd.NaT.__class__)):
        return str(obj) if not pd.isna(obj) else None
    else:
        return obj


def main():
    """Main function to process PR-related issues extraction"""
    log_activity("🚀 Starting PR Related Issues Extraction")

    # Load GitHub tokens
    try:
        all_tokens, token_cycle = load_tokens(TOKENS_FILE)
        log_activity(f"Loaded {len(all_tokens)} GitHub tokens")
    except Exception as e:
        log_activity(f"Error loading tokens: {e}")
        log_activity(f"Please ensure {TOKENS_FILE} exists and contains GitHub Personal Access Tokens.")
        return False

    # Load input CSV
    try:
        df = pd.read_csv(INPUT_CSV)
        log_activity(f"Loaded {len(df)} PRs from {INPUT_CSV}")
    except FileNotFoundError:
        log_activity(f"Error: {INPUT_CSV} not found.")
        return False
    except Exception as e:
        log_activity(f"Error loading CSV: {e}")
        return False

    # Check if issue_references column exists
    if "issue_references" not in df.columns:
        log_activity(f"Error: 'issue_references' column not found in {INPUT_CSV}")
        return False

    # Filter out rows with no issue references
    def has_valid_issues(issue_references_str):
        issue_numbers = parse_issue_numbers(issue_references_str)
        return len(issue_numbers) > 0
    
    df_with_valid_issues = df[df['issue_references'].apply(has_valid_issues)].copy()
    log_activity(f"Filtered to {len(df_with_valid_issues)} PRs with issue references (removed {len(df) - len(df_with_valid_issues)} PRs with no issue references)")

    if len(df_with_valid_issues) == 0:
        log_activity("No PRs with valid issues found. Exiting.")
        return False

    # Load previous progress
    progress_lock = threading.Lock()
    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as f:
            progress = pickle.load(f)
        log_activity(f"Loaded progress for {len(progress)} processed PRs.")
    else:
        progress = {}

    def process_pr(i, row):
        pr_id = row["id"]
        pull_number = row["pull_number"]
        issue_references_str = row["issue_references"]
        issue_numbers = parse_issue_numbers(issue_references_str)
        
        # Determine repository
        repository = row.get("repository_name_with_owner", DEFAULT_REPOSITORY)
        if pd.isna(repository):
            repository = DEFAULT_REPOSITORY

        if not issue_numbers:
            log_activity(
                f"[{i + 1}/{len(df_with_valid_issues)}] No valid issue numbers found for PR #{pull_number}, skipping"
            )
            return

        log_activity(
            f"[{i + 1}/{len(df_with_valid_issues)}] Processing PR #{pull_number} from {repository} with issues {issue_numbers}"
        )

        # Check if this PR has already been processed
        with progress_lock:
            if pr_id in progress:
                log_activity(f"[{i + 1}/{len(df_with_valid_issues)}] Skipping already processed PR {pr_id}")
                return

        # Collect all issue data for this PR
        all_issues_data = []
        
        # Process each issue number
        for issue_number in issue_numbers:
            log_activity(f"[{i + 1}/{len(df_with_valid_issues)}] Fetching issue #{issue_number} from {repository} ...")
            issue_data = fetch_issue(repository, issue_number, token_cycle, LOG_PATH)

            if issue_data:
                # Fetch comments
                comments_url = issue_data.get("comments_url")
                comments = []

                if comments_url:
                    log_activity(f"[{i + 1}/{len(df_with_valid_issues)}] Fetching comments for {repository}#{issue_number} ...")
                    comments = fetch_issue_comments(comments_url, token_cycle, LOG_PATH)
                    time.sleep(REQUEST_DELAY)

                all_issues_data.append({
                    "issue_number": issue_number,
                    "issue_id": issue_data.get("id"),
                    "issue_title": issue_data.get("title"),
                    "issue_body": issue_data.get("body"),
                    "issue_state": issue_data.get("state"),
                    "issue_created_at": issue_data.get("created_at"),
                    "issue_updated_at": issue_data.get("updated_at"),
                    "issue_closed_at": issue_data.get("closed_at"),
                    "issue_author": (
                        issue_data.get("user", {}).get("login")
                        if issue_data.get("user")
                        else None
                    ),
                    "issue_url": issue_data.get("html_url"),
                    "comments": comments
                })
            else:
                log_activity(f"⚠️ [{i + 1}/{len(df_with_valid_issues)}] Failed to fetch issue #{issue_number}")

            time.sleep(REQUEST_DELAY)

        if all_issues_data:
            # Combine all issue data into formatted strings
            issue_titles = []
            issue_bodies = []
            issue_states = []
            issue_authors = []
            issue_urls = []
            issue_ids = []
            issue_numbers_list = []
            issue_created_ats = []
            issue_updated_ats = []
            issue_closed_ats = []
            
            # Collect all comments with issue context
            all_formatted_comments = []
            total_comments_count = 0
            
            for issue_info in all_issues_data:
                issue_num = issue_info["issue_number"]
                issue_titles.append(f"issue #{issue_num}: {issue_info['issue_title'] or ''}")
                issue_bodies.append(f"issue #{issue_num}: {issue_info['issue_body'] or ''}")
                issue_states.append(f"issue #{issue_num}: {issue_info['issue_state'] or ''}")
                issue_authors.append(f"issue #{issue_num}: {issue_info['issue_author'] or ''}")
                issue_urls.append(f"issue #{issue_num}: {issue_info['issue_url'] or ''}")
                issue_ids.append(f"issue #{issue_num}: {issue_info['issue_id'] or ''}")
                issue_numbers_list.append(str(issue_num))
                issue_created_ats.append(f"issue #{issue_num}: {issue_info['issue_created_at'] or ''}")
                issue_updated_ats.append(f"issue #{issue_num}: {issue_info['issue_updated_at'] or ''}")
                issue_closed_ats.append(f"issue #{issue_num}: {issue_info['issue_closed_at'] or ''}")
                
                # Format comments with issue context
                if issue_info["comments"]:
                    formatted_comments = format_comments(issue_info["comments"], issue_num)
                    if formatted_comments:
                        all_formatted_comments.append(formatted_comments)
                    total_comments_count += len(issue_info["comments"])

            with progress_lock:
                # Store the processed issues data for this PR
                progress[pr_id] = {
                    # Original row data (all columns from input CSV)
                    "original_row_data": row.to_dict(),
                    # Combined issue-specific data
                    "repository": repository,
                    "issue_numbers": ",".join(issue_numbers_list),
                    "issue_ids": " \n\n ".join(issue_ids),
                    "issue_titles": " \n\n ".join(issue_titles),
                    "issue_bodies": " \n\n ".join(issue_bodies),
                    "issue_comments": "\n\n".join(all_formatted_comments),
                    "issue_comments_count": total_comments_count,
                    "issue_states": " \n\n ".join(issue_states),
                    "issue_created_ats": " \n\n ".join(issue_created_ats),
                    "issue_updated_ats": " \n\n ".join(issue_updated_ats),
                    "issue_closed_ats": " \n\n ".join(issue_closed_ats),
                    "issue_authors": " \n\n ".join(issue_authors),
                    "issue_urls": " \n\n ".join(issue_urls),
                }

                # Save progress after each PR
                with open(PICKLE_FILE, "wb") as f:
                    pickle.dump(progress, f)

            log_activity(f"✓ [{i + 1}/{len(df_with_valid_issues)}] Saved progress for PR {pr_id} ({len(all_issues_data)} issues, {total_comments_count} total comments)")
        else:
            log_activity(f"⚠️ [{i + 1}/{len(df_with_valid_issues)}] No issues fetched for PR {pr_id}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = [executor.submit(process_pr, i, row) for i, (_, row) in enumerate(df_with_valid_issues.iterrows())]
        # Wait for all futures to finish
        concurrent.futures.wait(futures)

    log_activity(f"✅ Done. Fetched data for {len(progress)} PRs with issues.")

    # Save to JSON
    clean_progress = clean_for_json(progress)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(clean_progress, f, indent=2)
    log_activity(f"📁 Saved JSON results to {OUTPUT_JSON}")

    # Save to CSV
    csv_rows = []
    for pr_id, issue_data in progress.items():
        # Get original row data
        original_data = issue_data.get("original_row_data", {})

        # Create a new row combining original data with issue data
        csv_row = original_data.copy()  # Start with all original columns

        # Add issue-specific columns (prefixed to avoid conflicts)
        csv_row.update({
            "issue_repository": issue_data.get("repository"),
            "issue_numbers": issue_data.get("issue_numbers"),
            "issue_ids": issue_data.get("issue_ids"),
            "issue_titles": issue_data.get("issue_titles"),
            "issue_bodies": issue_data.get("issue_bodies"),
            "issue_comments": issue_data.get("issue_comments"),
            "issue_comments_count": issue_data.get("issue_comments_count"),
            "issue_states": issue_data.get("issue_states"),
            "issue_created_ats": issue_data.get("issue_created_ats"),
            "issue_updated_ats": issue_data.get("issue_updated_ats"),
            "issue_closed_ats": issue_data.get("issue_closed_ats"),
            "issue_authors": issue_data.get("issue_authors"),
            "issue_urls": issue_data.get("issue_urls"),
        })

        csv_rows.append(csv_row)

    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)
    log_activity(f"📄 Saved CSV results to {OUTPUT_CSV}")
    
    log_activity(f"🎉 Successfully processed {len(progress)} PRs with related issues")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
