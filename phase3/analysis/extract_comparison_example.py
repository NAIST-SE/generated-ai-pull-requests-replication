import os
import sys
from pathlib import Path
import pandas as pd

# Add the parent directory of phase3 (i.e. repo root) to python path if not present
repo_root = Path(__file__).parent.parent.parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from phase3.data_loader import DataLoader

def main():
    # Initialize DataLoader using relative path from this script
    datasets_path = Path(__file__).parent.parent / "datasets"
    loader = DataLoader(str(datasets_path))
    
    print("Loading datasets...")
    df_human = loader.load_selected_prs()
    df_gen = loader.load_generated_descriptions()
    
    print(f"Human descriptions shape: {df_human.shape}")
    print(f"Generated descriptions shape: {df_gen.shape}")
    
    # We want to find PR #2213
    # Look up by pull_number
    pr_row = df_human[df_human['pull_number'] == 2213]
    if pr_row.empty:
        print("Error: PR #2213 not found in human descriptions.")
        return
        
    row = pr_row.iloc[0]
    pr_id = row['id']
    print("\n" + "="*80)
    print(f"PR Number: 2213")
    print(f"PR ID: {pr_id}")
    print(f"Title: {row.get('title')}")
    print(f"URL: {row.get('url')}")
    print("-"*80)
    print("HUMAN DESCRIPTION:")
    print(row.get('body'))
    print("-"*80)
    
    # Get generated description for this PR
    gen_rows = df_gen[df_gen['pr_id'] == pr_id]
    gpt5_p3 = gen_rows[(gen_rows['model'] == 'gpt-5-mini-2025-08-07') & (gen_rows['variation'] == 'P-3')]
    
    if not gpt5_p3.empty:
        print("AI GENERATED DESCRIPTION (gpt-5-mini, P-3 / File Diff):")
        print(gpt5_p3.iloc[0]['generated_text'])
    else:
        print("Warning: GPT-5-mini P-3 generated description not found for this PR.")
    print("="*80)

if __name__ == "__main__":
    main()
