import pandas as pd
from pathlib import Path
import sys

def merge_datasets():
    base_path = Path(__file__).parent.parent / "datasets"
    original_path = base_path / "jabref-prs.csv"
    additional_path = base_path / "jabref_prs_additional.csv"
    
    if not original_path.exists():
        print(f"Error: Original file not found at {original_path}")
        sys.exit(1)
        
    if not additional_path.exists():
        print(f"Error: Additional file not found at {additional_path}")
        sys.exit(1)
    
    print(f"Reading original dataset from {original_path}...")
    # Read original dataset
    df_original = pd.read_csv(original_path)
    
    # Handle unnamed index column if present
    if 'Unnamed: 0' in df_original.columns:
        print("Dropping 'Unnamed: 0' column from original dataset...")
        df_original = df_original.drop(columns=['Unnamed: 0'])
    elif df_original.columns[0] == '' or df_original.columns[0].startswith('Unnamed'):
         print(f"Dropping unnamed column '{df_original.columns[0]}' from original dataset...")
         df_original = df_original.drop(columns=[df_original.columns[0]])

        
    print(f"Original dataset shape: {df_original.shape}")
    
    print(f"Reading additional dataset from {additional_path}...")
    df_additional = pd.read_csv(additional_path)
    print(f"Additional dataset shape: {df_additional.shape}")
    
    # Rename columns in additional dataset to match original if needed
    # 'merge_commit_oid' -> 'merge_commit'
    if 'merge_commit_oid' in df_additional.columns:
        print("Renaming 'merge_commit_oid' to 'merge_commit' in additional dataset...")
        df_additional = df_additional.rename(columns={'merge_commit_oid': 'merge_commit'})
        
    # Concatenate datasets
    print("Concatenating datasets...")
    # Ensure we align columns. Columns not in one will be NaN in the other.
    df_merged = pd.concat([df_original, df_additional], ignore_index=True)
    print(f"Merged dataset shape (before deduplication): {df_merged.shape}")
    
    # Drop duplicates based on 'id'
    # We keep the first occurrence (original data) to preserve columns like 'repository' that might be missing in additional
    print("Dropping duplicates based on 'id'...")
    df_merged = df_merged.drop_duplicates(subset=['id'], keep='first')
    print(f"Merged dataset shape (after deduplication): {df_merged.shape}")
    
    # Save merged dataset
    print(f"Saving merged dataset to {original_path}...")
    df_merged.to_csv(original_path, index=False)
    print("Done!")

if __name__ == "__main__":
    merge_datasets()
