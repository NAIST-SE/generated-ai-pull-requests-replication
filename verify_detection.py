import pandas as pd
import os

FILE_PATH = "phase3/datasets/detection_results.csv"

def analyze_detection_results():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    print(f"Analyzing: {FILE_PATH}")
    try:
        df = pd.read_csv(FILE_PATH)
    except Exception as e:
        print(f"ERROR: Could not read file. {e}")
        return

    total_rows = len(df)
    print(f"Total Rows: {total_rows}")

    # Detectors based on columns
    detectors = set()
    for col in df.columns:
        if col.endswith("_success"):
            detector_name = col.replace("_success", "")
            detectors.add(detector_name)
    
    print(f"Detectors found: {sorted(list(detectors))}")

    for det in sorted(list(detectors)):
        success_col = f"{det}_success"
        prob_col = f"{det}_ai_probability"
        
        if success_col in df.columns:
            # Check boolean success
            # Sometimes CSV stores as string 'True'/'False' or 1/0
            # Pandas usually handles this, but let's be safe
            success_count = df[success_col].astype(bool).sum()

            fail_count = total_rows - success_count
            print(f"\n{det.upper()}:")
            print(f"  Success: {success_count}")
            print(f"  Failed:  {fail_count}")
            
            if prob_col in df.columns:
                null_probs = df[prob_col].isnull().sum()
                print(f"  Null Probability: {null_probs}")

    # Duplicates check
    if 'entry_key' in df.columns:
        dupes = df.duplicated(subset=['entry_key']).sum()
        if dupes > 0:
            print(f"\nWARNING: {dupes} duplicate entry_keys found!")
        else:
            print(f"\nNo duplicate entry_keys found.")
    else:
        # Fallback check
        if 'pr_id' in df.columns and 'variation' in df.columns and 'model' in df.columns:
             dupes = df.duplicated(subset=['pr_id', 'variation', 'model']).sum()
             if dupes > 0:
                 print(f"\nWARNING: {dupes} duplicate (pr_id, variation, model) found!")
    
analyze_detection_results()
