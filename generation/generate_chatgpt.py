"""
PR Description Generator for GPT-5-mini
Model: gpt-5-mini-2025-08-07 (or configurable)
Constraint: NO TRUNCATION.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import argparse
import logging
import datetime
import hashlib
import glob
import shutil
import math
import multiprocessing
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
import tiktoken

from generation.truncate_utils import truncate_pr_files, get_tokenizer_for_model
from generation.config import DECODING_PARAMS

# Try to import openai, handle failure gracefully
try:
    import openai
    from openai.types.chat import ChatCompletionMessageParam
except ImportError:
    openai = None
    ChatCompletionMessageParam = Any # type: ignore

# --- Configuration ---

# Model selection
DEFAULT_MODEL = "gpt-5-mini-2025-08-07" 

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "../datasets")
SELECTED_PRS_PATH = os.path.join(DATA_DIR, "selected_prs_sampled.csv")
PR_FILES_PATH = os.path.join(DATA_DIR, "selected_prs_with_diffs_and_file_contents.csv")
PR_ISSUES_PATH = os.path.join(DATA_DIR, "selected_prs_with_related_issues.csv")
OUTPUT_DIR = DATA_DIR

# Logging Setup
os.makedirs("logs", exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Ensure dir exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    handler = logging.FileHandler(log_file, mode='a')        
    handler.setFormatter(formatter)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)
    
    return logger

# Main logger
logger = setup_logger('main_logger', 'logs/generation_log.log')

# --- Data Structures ---

@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    model: str
    pr_id: str
    variation: str
    seed: int
    decoding_params: str  # JSON string (empty for API models)
    prompt_hash: str
    prompt_token_count: int
    context_token_count: int  # Same as prompt_token_count for ChatGPT
    generated_token_count: int
    full_prompt: str
    generated_text: str
    device_map: str = "api"  # Always "api" for ChatGPT
    finish_reason: str = "unknown"
    provider: str = "OpenAI"
    precision: str = "auto"
    truncated_token_count: Optional[int] = None  # Token count after truncation (if applicable)

# --- Prompt Variations Definition ---

@dataclass
class VariationConfig:
    id: str
    name: str
    include_repo_path: bool = False
    include_title: bool = False
    include_diffs: bool = False
    include_file_contents: bool = False
    include_issues: bool = False

VARIATIONS = {
    "P-1": VariationConfig("P-1", "Repo Path", include_repo_path=True),
    "P-2": VariationConfig("P-2", "PR Title", include_title=True),
    "P-3": VariationConfig("P-3", "PR Diffs", include_diffs=True),
    "P-4": VariationConfig("P-4", "File Contents", include_file_contents=True),
    "P-5": VariationConfig("P-5", "Related Issues", include_issues=True),
    "P-6": VariationConfig("P-6", "Without File Contents", 
                           include_repo_path=True, include_title=True, 
                           include_diffs=True, include_issues=True)
}

# --- Classes ---

class DataLoader:
    """Handles loading and merging of PR datasets."""
    
    @staticmethod
    def load_data() -> Dict[str, Dict[str, Any]]:
        """
        Loads all CSVs and returns a dictionary of PRs.
        Key: pr_id (str)
        Value: Dict containing 'info', 'files', 'issues'
        """
        logger.info("Loading datasets...")

        # 1. Load PR Files (Main content source)
        if not os.path.exists(PR_FILES_PATH):
            raise FileNotFoundError(f"Files CSV not found: {PR_FILES_PATH}")
        
        # Determine column types to avoid warnings
        logger.info(f"Reading from {PR_FILES_PATH}")
        df_files = pd.read_csv(PR_FILES_PATH)
        
        # 2. Load Issues (Context source)
        df_issues = pd.DataFrame()
        if os.path.exists(PR_ISSUES_PATH):
            logger.info(f"Reading from {PR_ISSUES_PATH}")
            df_issues = pd.read_csv(PR_ISSUES_PATH)
        else:
            logger.warning(f"Issues file not found: {PR_ISSUES_PATH}")
        
        # Group data by PR ID
        prs = {}
        
        # Process files
        logger.info(f"Processing {len(df_files)} file records...")
        for _, row in df_files.iterrows():
            # Use 'id' (GitHub ID) as primary key
            pr_id = str(row['id'])


            if pr_id not in prs:
                prs[pr_id] = {
                    'info': row.to_dict(), # Basic metadata from first file row
                    'files': [],
                    'issues': []
                }
            
            # Add file record
            file_record = {
                'filename': row['filename'],
                'patch': row['patch'] if pd.notna(row['patch']) else "",
                'file_content': row['file_content'] if pd.notna(row['file_content']) else "",
                'status': row['status'],
                'additions': row['additions'],
                'deletions': row['deletions']
            }
            prs[pr_id]['files'].append(file_record)
            
        # Process issues (Merge into PRS)
        if not df_issues.empty:
            logger.info("Merging issue data...")
            # We assume 'id' is the primary key and matches across datasets
            if 'id' in df_issues.columns:
                for _, row in df_issues.iterrows():
                    pr_id = str(row['id'])

                    if pr_id in prs:
                        # Append issue data. Avoid duplicates if possible
                        prs[pr_id]['issues'].append(row.to_dict())
            else:
                logger.warning("Issues CSV does not have 'id' column. Skipping issue merge.")

        logger.info(f"Loaded {len(prs)} unique PRs.")
        return prs

class PromptBuilder:
    """Builds prompts based on Variations without truncation."""
    
    @staticmethod
    def build_prompt(pr_data: Dict[str, Any], variation: VariationConfig) -> str:
        import math
        
        def safe_str(value, default=''):
            """Safely convert value to string, handling NaN and None."""
            if value is None:
                return default
            if isinstance(value, float) and math.isnan(value):
                return default
            return str(value)
        
        parts = []
        
        # 1. System/Task Instruction
        parts.append("You are a helpful assistant that generates GitHub pull request descriptions.")
        parts.append("Based on the provided context, write a clear and informative description.")
        parts.append('Respond with ONLY a JSON object in this format:\n{"description": "<your description here>"}\n')
        parts.append("Do not include any other text or explanation.")
        parts.append("### CONTEXT")

        files = pr_data['files']
        info = pr_data['info']
        # Issues might be in pr_data['issues'] or embedded in info depending on data source logic

        # 2. P-1: Repository & File Paths
        if variation.include_repo_path:
            repo_name = safe_str(info.get('repository_name_with_owner'), 'Unknown')
            parts.append(f"<repository>\n{repo_name}\n</repository>")
            
            file_paths = [safe_str(f.get('filename')) for f in files if safe_str(f.get('filename'))]
            if file_paths:
                parts.append("<affected_files>\n" + "\n".join(file_paths) + "\n</affected_files>")

        # 3. P-2: Title
        if variation.include_title:
            title = safe_str(info.get('title'))
            if title:
                parts.append(f"<title>\n{title}\n</title>")

        # 4. P-3: Diffs
        if variation.include_diffs:
            diff_text = []
            for f in files:
                fname = safe_str(f.get('filename'))
                patch = safe_str(f.get('patch'))
                if patch and fname:
                    diff_text.append(f"File: {fname}\n```diff\n{patch}\n```")
            if diff_text:
                parts.append("<diffs>\n" + "\n\n".join(diff_text) + "\n</diffs>")

        # 5. P-4: File Contents
        if variation.include_file_contents:
            # P-4 Logic: Add file contents (assumes upstream truncation if needed)
            content_text = []
            files_with_content = [f for f in files if safe_str(f.get('file_content'))]
            
            for f in files_with_content:
                fname = safe_str(f.get('filename'))
                content = safe_str(f.get('file_content'))
                if content and fname:
                    content_text.append(f"File: {fname}\n```\n{content}\n```")
            
            if content_text:
                parts.append("<files>\n" + "\n\n".join(content_text) + "\n</files>")

        # 6. P-5: Issues
        if variation.include_issues:
            # Check pr_data['issues'] first (from separate CSV)
            issues = pr_data.get('issues', [])
            
            issue_texts = []
            for issue in issues:
                title = safe_str(issue.get('title')) or safe_str(issue.get('issue_title'))
                body = safe_str(issue.get('body')) or safe_str(issue.get('issue_body'))
                if title or body:
                    issue_texts.append(f"Title: {title}\nBody: {body}")
            
            # Fallback to info dict if populated there (legacy)
            if not issue_texts:
                 i_titles = safe_str(info.get('issue_titles'))
                 i_bodies = safe_str(info.get('issue_bodies'))
                 if i_titles:
                     issue_texts.append(f"Titles: {i_titles}\nBodies: {i_bodies}")

            if issue_texts:
                parts.append("<issues>\n" + "\n---\n".join(issue_texts) + "\n</issues>")

        parts.append("### END CONTEXT")
        # Repeat instructions for long contexts
        parts.append("Reminder: based on the provided context, write a clear and informative description.")
        parts.append('Respond with ONLY a JSON object in this format:\n{"description": "<your description here>"}\n')
        parts.append("Do not include any other text or explanation.")

        return "\n\n".join(parts)

class Generator:
    """Manages OpenAI interactions."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, api_key: Optional[str] = None, mock: bool = False, temperature: Optional[float] = None):
        self.model_name = model_name
        self.mock = mock
        self.temperature = temperature
        self._check_api_key(api_key)

    def _check_api_key(self, api_key):
        if self.mock:
            self.client = None
            return

        if not openai:
            self.client = None
            return

        key = api_key or os.environ.get("OPENAI_API_KEY")
        # Also check local file if env var missing (common in this repo structure)
        if not key:
            try:
                # Look for env in typical locations
                possible_paths = [
                    "env/openai_api_key.txt",
                    "../env/openai_api_key.txt", 
                    "../../env/openai_api_key.txt"
                ]
                for p in possible_paths:
                    full_p = os.path.join(os.path.dirname(__file__), p)
                    if os.path.exists(full_p):
                        with open(full_p, 'r') as f:
                            key = f.read().strip()
                        break
            except Exception:
                pass
        
        if key:
            self.client = openai.OpenAI(api_key=key)
        else:
            self.client = None

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Executes the generation request.
        Returns dict with 'content', 'input_tokens', 'output_tokens', 'duration'
        """
        if not self.client:
            # Mock Response
            time.sleep(0.5)
            # Create a mock description based on prompt length or variation
            return {
                "content": json.dumps({"description": "MOCKED DESCRIPTION - API KEY MISSING"}),
                "input_tokens": 0,
                "output_tokens": 0,
                "duration": 500
            }

        start_time = time.time()
        try:
            # We use chat completion
            messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
            
            call_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},  # Enforce JSON
            }
            if self.temperature is not None:
                call_kwargs["temperature"] = self.temperature

            response = self.client.chat.completions.create(**call_kwargs
            )
            
            duration = int((time.time() - start_time) * 1000)
            
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason if response.choices else "unknown"

            return {
                "content": response.choices[0].message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "duration": duration
            }

        except Exception as e:
            # logger.error(f"Generation failed: {e}") # handled by outer loop
            return {
                "content": json.dumps({"description": "ERROR_GENERATION_FAILED"}),
                "error": str(e),
                "duration": int((time.time() - start_time) * 1000)
            }

# --- Parallel Implementation ---

def process_batch(
    tasks: List[Tuple[str, VariationConfig]], 
    all_prs: Dict[str, Any],
    generator: Generator, 
    args: argparse.Namespace, 
    exp_id: str,
    out_path: str,
    truncation_tokenizer: Any = None,
    encoder: Any = None
):
    """
    Process a batch of (pr_id, variation) tasks.
    """
    results = []
    
    for pr_id, var_config in tasks:
        pr_data = all_prs[pr_id]
        
        # Apply P-4 Truncation if enabled
        truncated_token_count = None
        current_pr_data = pr_data
        
        if var_config.id == "P-4" and args.truncate_p4 and truncation_tokenizer:
            import copy
            current_pr_data = copy.deepcopy(pr_data)
            files = current_pr_data.get('files', [])
            
            # Transform to format needed by truncate_utils
            files_for_truncation = []
            for f in files:
                content = f.get('file_content', '')
                if content:
                    files_for_truncation.append({
                        'path': f.get('filename', 'unknown'),
                        'content': content
                    })
            
            if files_for_truncation:
                truncated_files = truncate_pr_files(
                    files_for_truncation, 
                    max_total_tokens=args.max_context_tokens,
                    tokenizer=truncation_tokenizer,
                    preserve_ratio=0.4
                )
                
                # Update current_pr_data with truncated content
                truncation_map = {f['path']: f['content'] for f in truncated_files}
                for f in current_pr_data['files']:
                    fname = f.get('filename', 'unknown')
                    if fname in truncation_map:
                        f['file_content'] = truncation_map[fname]
                
                # Estimate truncated size for logging
                if hasattr(truncation_tokenizer, "encode"):
                    try:
                        truncated_token_count = sum(len(truncation_tokenizer.encode(f.get('content', ''))) for f in truncated_files)
                    except:
                        pass

        # Build Prompt
        prompt = PromptBuilder.build_prompt(current_pr_data, var_config)
        
        if encoder:
            try:
                tokens_approx = len(encoder.encode(prompt))
            except:
                tokens_approx = int(len(prompt) / 4)
        else:
            tokens_approx = int(len(prompt) / 4)
        
        if args.dry_run:
            # logger.info(f"Dry Run [PR {pr_id}, {var_config.id}]: ~{tokens_approx} tokens.")
            continue
        
        # Generate
        result = generator.generate(prompt)
        
        # Extract description
        try:
            content_json = json.loads(result.get("content", "{}"))
            desc_text = content_json.get("description", "")
        except json.JSONDecodeError:
            desc_text = result.get("content", "")
        
        # Create experiment record
        input_tokens = result.get("input_tokens", tokens_approx)
        output_tokens = result.get("output_tokens", 0)
        finish_reason = result.get("finish_reason", "unknown")
        
        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=datetime.datetime.now().isoformat(),
            model=args.model,
            pr_id=pr_id,
            variation=var_config.id,
            seed=args.seed,
            decoding_params=json.dumps({"temperature": args.temperature} if args.temperature is not None else {}),
            prompt_hash=hashlib.md5(prompt.encode('utf-8')).hexdigest(),
            prompt_token_count=input_tokens,
            context_token_count=input_tokens,
            generated_token_count=output_tokens,
            truncated_token_count=truncated_token_count,
            full_prompt=prompt,
            generated_text=desc_text,
            device_map="api",
            finish_reason=finish_reason,
        )
        results.append(asdict(record))
        
        # Sleep to respect rate limits
        time.sleep(1)

    # Save results if not empty
    if results:
        df = pd.DataFrame(results)
        # Check if file exists to append or write new
        # Note: In parallel mode, this function handles its own exclusive shard file, so direct write is fine.
        # But for robustness, we append.
        if os.path.exists(out_path):
            df.to_csv(out_path, mode='a', header=False, index=False)
        else:
            df.to_csv(out_path, index=False)

def run_worker(
    worker_id: int,
    tasks: List[Tuple[str, Any]], # (pr_id, var_config)
    all_prs: Dict[str, Any],
    args: argparse.Namespace,
    exp_id: str,
    base_out_path: str
):
    """
    Worker process for parallel generation.
    """
    # Setup worker logger
    log_file = f"logs/chatgpt_worker_{worker_id}.log"
    w_logger = setup_logger(f"worker_{worker_id}", log_file)
    w_logger.info(f"Worker {worker_id} started with {len(tasks)} tasks")
    
    # Construct shard path
    base_name, ext = os.path.splitext(base_out_path)
    shard_path = f"{base_name}_shard_{worker_id}{ext}"
    
    # Initialize generator
    generator = Generator(model_name=args.model, mock=args.mock, temperature=args.temperature)
    
    # Initialize tokenizers
    encoder = None
    try:
        encoder = tiktoken.encoding_for_model("gpt-4o")
    except:
        pass
        
    truncation_tokenizer = None
    if args.truncate_p4:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        truncation_tokenizer = get_tokenizer_for_model(args.model, hf_token=hf_token)

    # Write header if new file
    if not os.path.exists(shard_path):
        dummy = pd.DataFrame(columns=list(ExperimentRecord.__annotations__.keys()))
        dummy.to_csv(shard_path, index=False)

    # Batch process to save intermittently
    batch_size = 5
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        batch = tasks[i*batch_size : (i+1)*batch_size]
        w_logger.info(f"Processing batch {i+1}/{total_batches} ({len(batch)} items)")
        
        process_batch(
            batch, 
            all_prs, 
            generator, 
            args, 
            exp_id, 
            shard_path,
            truncation_tokenizer,
            encoder
        )
        
    w_logger.info("Worker finished successfully.")

def merge_shards(main_results_path: str, cleanup: bool = True):
    """
    Merge shard files into main results.
    """
    base_name, ext = os.path.splitext(main_results_path)
    pattern = f"{base_name}_shard_*{ext}"
    shard_files = glob.glob(pattern)
    
    if not shard_files:
        return

    print(f"Merging {len(shard_files)} shard files...")
    
    dfs = []
    # If main file exists, load it first
    if os.path.exists(main_results_path):
        try:
            dfs.append(pd.read_csv(main_results_path))
        except Exception:
            pass
            
    for f in shard_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading shard {f}: {e}")
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        # Deduplicate based on pr_id and variation
        if 'pr_id' in full_df.columns and 'variation' in full_df.columns:
            full_df = full_df.drop_duplicates(subset=['pr_id', 'variation'], keep='last')
        
        full_df.to_csv(main_results_path, index=False)
        print(f"Merged results saved to {main_results_path} ({len(full_df)} rows)")
        
        if cleanup:
            for f in shard_files:
                os.remove(f)
                print(f"Removed {f}")

def cleanup_duplicates(out_path: str):
    """
    Final cleanup to remove duplicates from the results file.
    """
    if not os.path.exists(out_path):
        return
        
    try:
        df = pd.read_csv(out_path)
        if 'pr_id' in df.columns and 'variation' in df.columns:
            initial_len = len(df)
            df = df.drop_duplicates(subset=['pr_id', 'variation'], keep='last')
            if len(df) < initial_len:
                print(f"Removed {initial_len - len(df)} duplicate rows.")
                df.to_csv(out_path, index=False)
    except Exception as e:
        print(f"Warning during cleanup: {e}")

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Generate PR Descriptions (No Truncation)")
    parser.add_argument("--variation", type=str, choices=list(VARIATIONS.keys()) + ["ALL"], required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI Model ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit", type=int, help="Limit number of PRs to process")
    parser.add_argument("--dry-run", action="store_true", help="Do not call API")
    parser.add_argument("--mock", action="store_true", help="Force mock generation")
    parser.add_argument("--truncate-p4", action="store_true", help="Enable P-4 truncation")
    parser.add_argument("--max-context-tokens", type=int, default=30000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to main results CSV to resume from")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")

    args = parser.parse_args()
    
    # 1. Load Data
    loader = DataLoader()
    all_prs = loader.load_data()
    
    if args.limit:
        logger.info(f"Limiting to first {args.limit} PRs")
        # Ensure consistent order for limit
        keys = list(all_prs.keys())[:args.limit]
        all_prs = {k: all_prs[k] for k in keys}
    
    # 2. Determine Variations and Output Path
    vars_to_run = [VARIATIONS[args.variation]] if args.variation != "ALL" else list(VARIATIONS.values())
    
    exp_id = f"exp_chatgpt_{int(time.time())}"
    
    # Determine base output path
    if args.resume:
        out_path = args.resume
        # Try to extract exp_id from filename if possible, otherwise keep generated
    else:
        out_path = os.path.join(OUTPUT_DIR, f"chatgpt_results_{exp_id}.csv")

    # 3. Startup Recovery: Merge any existing shards from previous run
    merge_shards(out_path, cleanup=True)
    
    # 4. Check for completed work
    completed_pairs = set()
    if os.path.exists(out_path):
        try:
            df = pd.read_csv(out_path)
            # Re-read exp_id if available
            if 'experiment_id' in df.columns and not df.empty:
                exp_id = str(df.iloc[0]['experiment_id'])
            
            for _, row in df.iterrows():
                completed_pairs.add((str(row['pr_id']), row['variation']))
            print(f"Found {len(completed_pairs)} already completed tasks.")
        except Exception as e:
            print(f"Warning reading existing file: {e}")

    # 5. Generate Task List
    all_tasks = []
    for pr_id in all_prs:
        for var in vars_to_run:
            if (str(pr_id), var.id) not in completed_pairs:
                all_tasks.append((pr_id, var))
    
    print(f"To process: {len(all_tasks)} tasks "
          f"(Total requested: {len(all_prs) * len(vars_to_run)})")

    if not all_tasks:
        print("All tasks completed.")
        return

    # 6. Execution
    if args.workers > 1:
        print(f"Spawning {args.workers} workers...")
        
        # Split tasks
        chunk_size = math.ceil(len(all_tasks) / args.workers)
        chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
        
        processes = []
        for i, chunk in enumerate(chunks):
            p = multiprocessing.Process(
                target=run_worker,
                args=(i, chunk, all_prs, args, exp_id, out_path)
            )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
            
        print("All workers finished. Merging results...")
        merge_shards(out_path, cleanup=True)
        
    else:
        # Sequential Execution (original behavior wrapped)
        print("Running sequentially...")
        
        # Setup generator for main process
        generator = Generator(model_name=args.model, mock=args.mock, temperature=args.temperature)
        
        # We reuse process_batch but just call it directly
        # Init tokenizers
        encoder = None
        try: 
            encoder = tiktoken.encoding_for_model("gpt-4o")
        except: 
            pass
            
        truncation_tokenizer = None
        if args.truncate_p4:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            truncation_tokenizer = get_tokenizer_for_model(args.model, hf_token=hf_token)
            
        # We can treat this as "Shard -1" or just write direct
        # Let's use process_batch writing to out_path directly
        
        # Batching for sequential
        batch_size = 10
        total_batches = (len(all_tasks) + batch_size - 1) // batch_size
        
        for i in range(total_batches):
            batch = all_tasks[i*batch_size : (i+1)*batch_size]
            print(f"Processing batch {i+1}/{total_batches}")
            process_batch(
                batch,
                all_prs,
                generator,
                args,
                exp_id,
                out_path,
                truncation_tokenizer,
                encoder
            )

    cleanup_duplicates(out_path)
    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # Safe for macOS/Torch/OpenAI
    main()
