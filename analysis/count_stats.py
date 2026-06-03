import pandas as pd
import json
import re
from pathlib import Path

# Paths
DATASETS_DIR = Path("../datasets")
FILES_CSV = DATASETS_DIR / "selected_prs_with_diffs_and_file_contents.csv"
ISSUES_JSON = DATASETS_DIR / "selected_prs_with_related_issues.json"

def count_stats():
    # 1. Analyze Files and Diffs
    print(f"Loading {FILES_CSV}...")
    try:
        df_files = pd.read_csv(FILES_CSV)
        
        # Count Files
        total_files = len(df_files)
        
        # Count Diff Hunks
        # Hunks usually start with @@ -x,y +a,b @@
        # We'll count occurrence of this pattern in the patch column
        hunk_pattern = re.compile(r'^@@\s-[0-9,]+\s\+[0-9,]+\s@@', re.MULTILINE)
        
        total_hunks = 0
        for patch in df_files['patch'].dropna():
            matches = hunk_pattern.findall(patch)
            total_hunks += len(matches)
            
        print(f"Total Files: {total_files}")
        print(f"Total Diff Hunks: {total_hunks}")
        
    except Exception as e:
        print(f"Error processing files CSV: {e}")

    # 2. Analyze Unique Linked Issues
    print(f"\nLoading {ISSUES_JSON}...")
    try:
        with open(ISSUES_JSON, 'r') as f:
            data = json.load(f)
            
        unique_issues = set()
        
        for pr_id, pr_data in data.items():
            # Data seems to use flat keys like 'issue_numbers' which is a comma-separated string
            issue_nums_str = pr_data.get('issue_numbers')
            
            if issue_nums_str:
                # Assuming issue_numbers depends on repository, but helper likely ran for one repo
                # For safety, let's keep just the number for now as we are likely in one repo context
                parts = str(issue_nums_str).split(',')
                for p in parts:
                    p = p.strip()
                    if p:
                        unique_issues.add(p)

        print(f"Total Unique Linked Issues: {len(unique_issues)}")
        
    except Exception as e:
        print(f"Error processing issues JSON: {e}")

if __name__ == "__main__":
    count_stats()
