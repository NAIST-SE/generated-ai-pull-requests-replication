"""
Detection Pipeline

This script runs AI detection on both Human and AI-generated PR descriptions.
It uses the common detection tools and utilities.

Input:
    - Human Baseline: datasets/selected_prs_sampled.csv
    - AI Generated: datasets/generated_descriptions.csv

Output:
    - Results: datasets/detection_results.csv
"""

import sys
import os
from pathlib import Path
import pandas as pd
import argparse
import time
import random
import multiprocessing
import math
import shutil
import glob
from typing import List, Dict, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_loader import DataLoader
from detection_tools import get_available_detectors, create_detector
from detection_utils import DetectionLogger, ProgressTracker, CSVHandler

# Constants
DETECTION_RESULTS_PATH = "datasets/detection_results.csv"
LOG_DIR = "detection/logs"
PROGRESS_DIR = "detection/progress"
ZERO_GPT_TOKENS_FILE = "detection/env/zero-gpt-tokens.txt" # Path relative to project root
GPTZERO_API_KEY_FILE = "detection/env/gptzero-api-key.txt" # Path relative to project root

def prepare_entries(loader: DataLoader) -> List[Dict[str, Any]]:
    """
    Load data and prepare detection entries.
    """
    entries = []
    
    # 1. Load AI Generated
    print("Loading AI Generated descriptions...")
    df_ai = loader.load_generated_descriptions()
    
    generated_pr_ids = set()
    if df_ai.empty:
        print("Warning: No generated descriptions found.")
    else:
        generated_pr_ids = set(df_ai['pr_id'].astype(str))
        print(f"Found generated descriptions for {len(generated_pr_ids)} unique PRs.")
    
    # 2. Load Human Baseline
    print("Loading Human Baseline...")
    df_human = loader.load_selected_prs()
    human_entry_count = 0
    
    for _, row in df_human.iterrows():
        # Clean the body text just in case
        body = row.get('body', '')
        if pd.isna(body):
            body = ""
        body = str(body)
        
        if len(body.strip()) == 0:
            continue
            
        pr_id = str(row['id'])
        
        # Check if this PR has a generated counterpart
        if pr_id not in generated_pr_ids:
            continue
            
        entries.append({
            "pr_id": str(row['id']),
            "text": body,
            "type": "human",
            "variation": "original",
            "model": "human",
            "entry_key": f"{row['id']}_human"
        })
        human_entry_count += 1
        
    print(f"Loaded {human_entry_count} human entries (filtered from {len(df_human)} total).")

    # 3. Add AI generated entries
    if not df_ai.empty:
        for _, row in df_ai.iterrows():
            # Handle potential column name differences
            if 'generated_description' in row:
                body = row.get('generated_description', '')
            else:
                body = row.get('generated_text', '')
                
            if pd.isna(body):
                body = ""
            body = str(body)
            
            if len(body.strip()) == 0:
                continue

            # Skip generation errors (like exceeded tokens limit)
            if "ERROR_GENERATION_FAILED" in body:
                print(f"Skipping failed generation for PR {row.get('pr_id')}, variation {row.get('variation')}")
                continue

            pr_id = str(row['pr_id'])
            variation = str(row['variation'])
            model = str(row['model'])
            
            # Construct a safe key
            safe_model = model.replace(" ", "_").replace("/", "_")
            key = f"{pr_id}_{variation}_{safe_model}"
            
            entries.append({
                "pr_id": pr_id,
                "text": body,
                "type": "ai",
                "variation": variation,
                "model": model,
                "entry_key": key
            })
            
    print(f"Total entries to process: {len(entries)}")
    return entries

def process_with_detector(
    detector,
    entries: List[Dict[str, Any]],
    batch_size: int,
    logger: DetectionLogger,
    progress_tracker: ProgressTracker,
    csv_handler: CSVHandler,
    limit: int = 0,
    max_retries: int = 5,
    initial_retry_wait: int = 2
):
    """
    Process entries with a single detector using batching and resume capability.
    """
    detector_name = detector.get_name()
    logger.log("INFO", detector_name, f"Starting detection with {detector.get_model_version()}")
    
    # Filter out already processed entries
    unprocessed_entries = progress_tracker.get_unprocessed_entries(detector_name, entries)
    logger.log("INFO", detector_name, f"Found {len(unprocessed_entries)} unprocessed entries out of {len(entries)} total")
    
    if not unprocessed_entries:
        logger.log("SUCCESS", detector_name, "All entries already processed!")
        return

    # Apply limit if specified
    if limit > 0:
        unprocessed_entries = unprocessed_entries[:limit]
        logger.log("INFO", detector_name, f"Limiting processing to {limit} entries")

    # Process in batches
    total_batches = (len(unprocessed_entries) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(unprocessed_entries))
        batch = unprocessed_entries[batch_start:batch_end]
        
        batch_results = []
        batch_start_time = time.time()
        
        logger.log("INFO", detector_name, f"Processing batch {i+1}/{total_batches} ({len(batch)} entries)")
        
        for entry in batch:
            entry_start_time = time.time()
            text = entry["text"]
            
            result = None
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # SAFEGUARD: Truncate excessively long text (e.g. from repetition loops)
                    # 15000 chars is roughly 3000-4000 words, plenty for detection but safe from 16k token loops
                    if len(text) > 15000:
                        logger.log("WARNING", detector_name, f"Truncating text for {entry['entry_key']} (Length: {len(text)} chars)")
                        text = text[:15000]

                    result = detector.detect(text)
                    
                    # Check for explicit failure in result
                    if not result.get("success", False):
                        error_msg = result.get("error", "Unknown error")
                        # If error is about rate limiting (429), timeouts, or server errors (500), we retry
                        error_str = str(error_msg).lower()
                        if "429" in error_str or "500" in error_str or "timeout" in error_str or "try again" in error_str:
                             raise Exception(f"Retryable error: {error_msg}")
                    
                    break # Success
                except Exception as e:
                    retry_count += 1
                    wait_time = initial_retry_wait * (2 ** (retry_count - 1)) # Exponential backoff
                    logger.log("WARNING", detector_name, f"Error processing entry {entry['entry_key']}: {e}. Retrying in {wait_time}s... ({retry_count}/{max_retries})")
                    time.sleep(wait_time)
            
            if result is None:
                # Failed after retries
                logger.log("ERROR", detector_name, f"Failed to process entry {entry['entry_key']} after {max_retries} retries")
                result = {
                    "detector": detector_name,
                    "model_version": detector.get_model_version(),
                    "success": False,
                    "error": "Max retries exceeded",
                    "ai_probability": 0.0,
                    "is_ai_generated": False, 
                    "raw_response": "{}"
                }
            
            # Combine raw entry data with result data
            # Be careful not to overwrite key fields
            combined = entry.copy()
            combined.update(result)
            batch_results.append(combined)
            
            # Log individual performance
            logger.log_performance(detector_name, "entry_time", time.time() - entry_start_time)

        # Batch complete
        batch_time = time.time() - batch_start_time
        successful_count = sum(1 for r in batch_results if r.get("success", False))
        failed_count = len(batch_results) - successful_count
        
        logger.log_batch_summary(detector_name, i+1, len(batch), successful_count, failed_count, batch_time)
        
        # Save progress and CSV
        # 1. Update master CSV
        df = csv_handler.load_or_create_df()
        
        # Ensure base columns exist
        for col in ["pr_id", "variation", "model", "type", "text", "entry_key"]:
            if col not in df.columns:
                df[col] = None

        # Add base info for new entries if not present
        existing_keys = set(df['entry_key'])
        new_rows = []
        for res in batch_results:
            if res['entry_key'] not in existing_keys:
                new_rows.append({
                    "pr_id": res['pr_id'],
                    "variation": res['variation'],
                    "model": res['model'],
                    "entry_type": res['type'], # Map 'type' to 'entry_type' for consistency
                    "entry_key": res['entry_key'],
                    "text": res['text']
                })
        
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Merge results columns
        df = csv_handler.merge_detector_results(df, detector_name, batch_results)
        csv_handler.save_df(df)
        
        # 2. Save checkpoint (Success tracking)
        # Load existing progress first to not overwrite previous batches
        existing_df, processed_keys, _ = progress_tracker.load_progress(detector_name)
        
        # Update processed keys
        processed_keys.update(r['entry_key'] for r in batch_results)
        
        # Save
        # We don't necessarily need to save the full df in pickle if we have the CSV, 
        # but the progress tracker expects a DF. Let's save a light version.
        checkpoint_df = pd.DataFrame(batch_results) 
        if not existing_df.empty:
            checkpoint_df = pd.concat([existing_df, checkpoint_df], ignore_index=True)
            
        progress_tracker.save_progress(detector_name, checkpoint_df, processed_keys)
        
        # Sleep between batches to be nice to APIs
        time.sleep(1)

def run_worker(
    worker_id: int,
    entries: List[Dict[str, Any]],
    detectors_list: List[str],
    project_root: str,
    base_log_dir: str,
    base_progress_dir: str,
    base_results_path: str,
    batch_size: int,
    limit: int = 0,
    gptzero_key_file: str = None,
    tokens_file: str = None
):
    """
    Worker process for parallel detection.
    """
    # Setup worker-specific paths
    worker_log_dir = os.path.join(base_log_dir, f"shard_{worker_id}")
    worker_progress_dir = os.path.join(base_progress_dir, f"shard_{worker_id}")
    
    # Construct shard CSV path: detection_results_shard_0.csv
    base_name, ext = os.path.splitext(base_results_path)
    worker_results_path = f"{base_name}_shard_{worker_id}{ext}"
    
    # Initialize worker-specific tools
    logger = DetectionLogger(worker_log_dir, master_log_name=f"master_shard_{worker_id}.log")
    progress_tracker = ProgressTracker(worker_progress_dir)
    csv_handler = CSVHandler(worker_results_path)
    
    logger.log("INFO", f"worker_{worker_id}", f"Worker {worker_id} started with {len(entries)} entries")
    
    # Run Detectors
    for detector_name in detectors_list:
        logger.log("INFO", f"worker_{worker_id}", f"Initializing detector: {detector_name}")
        
        try:
            kwargs = {}
            if detector_name == "gptzero":
                if gptzero_key_file and os.path.exists(gptzero_key_file):
                    with open(gptzero_key_file, 'r') as f:
                        kwargs["api_key"] = f.read().strip()
                else:
                    logger.log("ERROR", f"worker_{worker_id}", "GPTZero API Key file not found")
                    continue
            elif detector_name == "zerogpt":
                 if tokens_file:
                    kwargs["tokens_file"] = tokens_file
            
            detector = create_detector(detector_name, **kwargs)
            
            process_with_detector(
                detector,
                entries,
                batch_size,
                logger,
                progress_tracker,
                csv_handler,
                limit=limit
            )
            
        except Exception as e:
            logger.log("ERROR", f"worker_{worker_id}", f"Failed to run detector {detector_name}: {e}")
            import traceback
            traceback.print_exc()
            
    logger.log("SUCCESS", f"worker_{worker_id}", f"Worker {worker_id} finished")


def merge_shards(
    main_results_path: str,
    main_progress_dir: str,
    main_log_dir: str,
    num_shards: int = -1,  # If -1, look for all matching files
    cleanup: bool = True
):
    """
    Merge shard results and progress into main files.
    """
    print("Merging shard results...")
    
    main_csv_handler = CSVHandler(main_results_path)
    main_df = main_csv_handler.load_or_create_df()
    
    # Find all shard CSVs
    base_name, ext = os.path.splitext(main_results_path)
    if num_shards > 0:
        shard_files = [f"{base_name}_shard_{i}{ext}" for i in range(num_shards)]
    else:
        pattern = f"{base_name}_shard_*{ext}"
        shard_files = glob.glob(pattern)
        
    merged_count = 0
    for shard_file in shard_files:
        if not os.path.exists(shard_file):
            continue
            
        print(f"Merging {shard_file}...")
        try:
            shard_df = pd.read_csv(shard_file)
            if not shard_df.empty:
                # We can just concat and let deduplication handle it, OR match columns
                # The CSVHandler structure implies we should be careful about merging
                # But since all shards have same structure (dynamically added cols), concat should work
                # provided we align columns.
                
                # Align columns
                for col in shard_df.columns:
                    if col not in main_df.columns:
                        main_df[col] = None
                
                main_df = pd.concat([main_df, shard_df], ignore_index=True)
                merged_count += 1
        except Exception as e:
            print(f"Error merging {shard_file}: {e}")
            
    if merged_count > 0:
        main_csv_handler.save_df(main_df, deduplicate=True)
        print("Merged and saved CSV results.")
    
    # Merge Progress (a bit trickier, but mostly union of processed sets)
    # We mainly need to update the processed sets for future runs
    # But ProgressTracker stores per detector.
    
    detectors = get_available_detectors()
    main_tracker = ProgressTracker(main_progress_dir)
    
    for detector in detectors:
        # Load main
        main_prog_df, main_processed, main_meta = main_tracker.load_progress(detector)
        updated = False
        
        # Look for shards for this detector
        # Shards are in subdirectories: progress_dir/shard_N/detection_progress_{detector}.pkl
        
        # We need to find all shard directories
        shard_dirs = glob.glob(os.path.join(main_progress_dir, "shard_*"))
        
        for shard_dir in shard_dirs:
            shard_tracker = ProgressTracker(shard_dir)
            s_df, s_processed, _ = shard_tracker.load_progress(detector)
            
            if s_processed:
                main_processed.update(s_processed)
                if not s_df.empty:
                    main_prog_df = pd.concat([main_prog_df, s_df], ignore_index=True)
                updated = True
                
        if updated:
            # Deduplicate by entry_key in df if exists
            if not main_prog_df.empty and 'entry_key' in main_prog_df.columns:
               main_prog_df = main_prog_df.drop_duplicates(subset='entry_key', keep='last')
            
            main_tracker.save_progress(detector, main_prog_df, main_processed, main_meta)
            print(f"Updated progress for {detector}")

    # Cleanup
    if cleanup:
        print("Cleaning up shard files...")
        for shard_file in shard_files:
            if os.path.exists(shard_file):
                os.remove(shard_file)
                
        shard_dirs = glob.glob(os.path.join(main_progress_dir, "shard_*"))
        for d in shard_dirs:
            shutil.rmtree(d, ignore_errors=True)
            
        shard_log_dirs = glob.glob(os.path.join(main_log_dir, "shard_*"))
        for d in shard_log_dirs:
            shutil.rmtree(d, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="AI Detection Pipeline")
    parser.add_argument("--detectors", nargs="+", default=["zerogpt", "gptzero", "roberta", "roberta-qa"],
                        help="List of detectors to run")
    parser.add_argument("--batch-size", type=int, default=10,  
                        help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of entries to process (0 for all)")
    parser.add_argument("--force", action="store_true", 
                        help="Force reprocessing of all entries")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1)")
    args = parser.parse_args()

    # Paths
    output_csv_abs = os.path.abspath(os.path.join(os.getcwd(), DETECTION_RESULTS_PATH))
    tokens_file_abs = os.path.abspath(os.path.join(os.getcwd(), ZERO_GPT_TOKENS_FILE))
    gptzero_key_file_abs = os.path.abspath(os.path.join(os.getcwd(), GPTZERO_API_KEY_FILE))
    
    # 1. Recovery Step
    # Check for any leftover shards from previous runs and merge them
    # This prevents lost work if a parallel run crashes
    merge_shards(output_csv_abs, PROGRESS_DIR, LOG_DIR, num_shards=-1, cleanup=True)

    # Setup Main Tools
    logger = DetectionLogger(LOG_DIR)
    progress_tracker = ProgressTracker(PROGRESS_DIR)
    csv_handler = CSVHandler(output_csv_abs)
    
    logger.log("INFO", "system", "Starting Detection Pipeline")
    
    # 2. Load Data
    loader = DataLoader()  # Assumes running from project root
    entries = prepare_entries(loader)
    
    if not entries:
        logger.log("ERROR", "system", "No entries found to process. Exiting.")
        return

    # Shuffle entries
    random.seed(42)
    random.shuffle(entries)

    # Clear progress if forced
    if args.force:
        logger.log("WARNING", "system", "Force flag set. Clearing all progress!")
        for det in args.detectors:
            progress_tracker.clear_progress(det)
            
    # Filter out already processed entries
    # We check against ALL detectors. If an entry is processed by ALL requested detectors, skip it?
    # Or just let individual detectors handle skipping?
    # To utilize workers efficiently, we should filter entries that are done for ALL requested detectors.
    # However, different detectors might be in different states.
    # A simple approach: Let workers handle the skipping (using process_with_detector's logic).
    # BUT, if we split the list, we might give a worker only processed entries.
    # Optimization: Pre-filter entries that are completed for ALL requested detectors.
    
    entries_to_process = []
    
    entries_to_process = []
    
    # Load successful keys per detector
    successful_keys_sets = []
    for det in args.detectors:
        df, _, _ = progress_tracker.load_progress(det)
        if not df.empty and "success" in df.columns:
            # Be robust about column existence and types
            s_keys = set(df[df["success"] == True]["entry_key"])
        else:
            s_keys = set()
        successful_keys_sets.append(s_keys)
        
    for entry in entries:
        key = entry['entry_key']
        # Check if key is in ALL successful sets
        all_done = all(key in s_keys for s_keys in successful_keys_sets)
        if not all_done:
            entries_to_process.append(entry)
            
    logger.log("INFO", "system", f"Filtered {len(entries) - len(entries_to_process)} fully completed entries.")
    logger.log("INFO", "system", f"Remaining entries to process: {len(entries_to_process)}")
    
    if not entries_to_process:
        logger.log("SUCCESS", "system", "All entries processed for all requested detectors!")
        return

    # Limit if requested
    if args.limit > 0:
        entries_to_process = entries_to_process[:args.limit]
        logger.log("INFO", "system", f"Limiting global processing to {len(entries_to_process)} entries")

    # Execution Mode
    if args.workers > 1:
        logger.log("INFO", "system", f"Running in PARALLEL mode with {args.workers} workers")
        
        # Split entries
        chunk_size = math.ceil(len(entries_to_process) / args.workers)
        chunks = [entries_to_process[i:i + chunk_size] for i in range(0, len(entries_to_process), chunk_size)]
        
        # Guard against fewer chunks than workers
        if len(chunks) < args.workers:
            logger.log("WARNING", "system", f"Not enough data for {args.workers} workers. Reducing to {len(chunks)}.")
        
        processes = []
        for i, chunk in enumerate(chunks):
            p = multiprocessing.Process(
                target=run_worker,
                args=(
                    i, 
                    chunk, 
                    args.detectors, 
                    os.getcwd(), 
                    LOG_DIR, 
                    PROGRESS_DIR, 
                    output_csv_abs, 
                    args.batch_size, 
                    0, # No limit per worker, we limited globally
                    gptzero_key_file_abs,
                    tokens_file_abs
                )
            )
            processes.append(p)
            p.start()
            
        # Wait for all
        for p in processes:
            p.join()
            
        logger.log("SUCCESS", "system", "All workers finished.")
        
        # Merge results
        merge_shards(output_csv_abs, PROGRESS_DIR, LOG_DIR, num_shards=len(processes), cleanup=True)
        
    else:
        logger.log("INFO", "system", "Running in SEQUENTIAL mode")
        
        # Standard sequential run
        for detector_name in args.detectors:
            logger.log("INFO", "system", f"Initializing detector: {detector_name}")
            
            try:
                kwargs = {}
                if detector_name == "gptzero":
                    if os.path.exists(gptzero_key_file_abs):
                        with open(gptzero_key_file_abs, 'r') as f:
                            kwargs["api_key"] = f.read().strip()
                    else:
                        logger.log("ERROR", "system", f"GPTZero API Key file not found at {gptzero_key_file_abs}")
                        continue
                elif detector_name == "zerogpt":
                     kwargs["tokens_file"] = tokens_file_abs
                
                detector = create_detector(detector_name, **kwargs)
                
                process_with_detector(
                    detector,
                    entries_to_process, # Use filtered list
                    args.batch_size,
                    logger,
                    progress_tracker,
                    csv_handler,
                    limit=0 # Already limited
                )
                
            except Exception as e:
                logger.log("ERROR", "system", f"Failed to run detector {detector_name}: {e}")
                import traceback
                traceback.print_exc()

    logger.log("SUCCESS", "system", "Detection pipeline completed.")

if __name__ == "__main__":
    main()
