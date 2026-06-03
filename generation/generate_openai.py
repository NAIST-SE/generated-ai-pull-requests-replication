"""
Unified OpenAI-compatible runner for Llama and Qwen generation.

This merges the overlapping logic from `generate_llama_openai.py` and
`generate_qwen_openai.py` into a single entrypoint. Choose the prompt
builder via `--builder` (llama|qwen|auto). When `auto` the builder is
inferred from the `--model` name.

Supports `--dry-run` to validate prompt construction without network calls.
"""

import os
import sys
import json
import time
import argparse
import logging
import hashlib
import glob
import shutil
import math
import multiprocessing
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import DataLoader, CONTEXT_MAPPING
from generation.config import DECODING_PARAMS

# Import both prompt builders and select at runtime
from generation.generate_llama import PromptBuilder as LlamaPromptBuilder
from generation.generate_qwen import PromptBuilder as QwenPromptBuilder
from generation.truncate_utils import truncate_pr_files, get_tokenizer_for_model

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
    provider: str = "auto"
    precision: str = "auto"
    truncated_token_count: Optional[int] = None

LOG_DIR = os.path.join(os.path.dirname(__file__), "../datasets")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup logger"""
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
logger = setup_logger('main_logger', os.path.join(LOG_DIR, "openai_generation.log"))


def create_client(base_url: str, api_key: Optional[str] = None) -> Any:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI Python client is required for actual generation: install openai-sdk or equivalent") from e

    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
        if "openrouter.ai" in base_url:
            # OpenRouter recommends these headers for rankings/stats
            client_kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/iwadi/llm-in-prs",
                "X-Title": "LLM in PRs Research"
            }
        elif "huggingface.co" in base_url:
            # Hugging Face router headers (optional but recommended)
            client_kwargs["default_headers"] = {
                "X-Source": "llm-in-prs-research"
            }

    if api_key:
        client_kwargs["api_key"] = api_key

    client = OpenAI(**client_kwargs)
    return client


def select_prompt_builder(builder_arg: str, model_name: str):
    b = builder_arg.lower()
    if b == "auto":
        if "qwen" in model_name.lower():
            return QwenPromptBuilder
        return LlamaPromptBuilder
    if b in ("qwen", "qwen_prompt", "qwenbuilder"):
        return QwenPromptBuilder
    return LlamaPromptBuilder


def build_messages_for(builder_cls, pr_data: Dict[str, Any], variation: str, 
                       truncate_p4: bool = False, tokenizer = None, max_tokens: int = 30000) -> tuple[List[Dict[str, str]], Optional[int]]:
    """
    Build chat messages for a PR using the specified builder.
    """
    truncated_count = None
    # Apply truncation for P-4 if enabled
    if variation == "P-4" and truncate_p4 and tokenizer:
        pr_data, truncated_count = apply_p4_truncation(pr_data, tokenizer, max_tokens)
    
    ctx = builder_cls.construct_context_block(pr_data, variation)
    return builder_cls.wrap_context_in_prompt(ctx), truncated_count


def apply_p4_truncation(pr_data: Dict[str, Any], tokenizer, max_tokens: int) -> tuple[Dict[str, Any], int]:
    """
    Apply intelligent truncation to P-4 file contents.
    """
    import copy
    pr_data_copy = copy.deepcopy(pr_data)
    
    files = pr_data_copy.get('files', [])
    
    # Prepare files for truncation utility
    files_for_truncation = []
    for f in files:
        content = f.get('file_content', '')
        if content:
            files_for_truncation.append({
                'path': f.get('filename', 'unknown'),
                'content': content
            })
    
    if not files_for_truncation:
        return pr_data_copy, 0
    
    # Apply truncation
    truncated_files = truncate_pr_files(
        files_for_truncation,
        max_total_tokens=max_tokens,
        tokenizer=tokenizer,
        preserve_ratio=0.4
    )

    # Calculate tokens of truncated content
    truncated_count = 0
    try:
        if hasattr(tokenizer, "encode"):
             truncated_count = sum(len(tokenizer.encode(f.get('content', ''))) for f in truncated_files)
    except Exception:
        pass
    
    # Update pr_data with truncated content
    truncation_map = {f['path']: f['content'] for f in truncated_files}
    
    for f in pr_data_copy['files']:
        fname = f.get('filename', 'unknown')
        if fname in truncation_map:
            f['file_content'] = truncation_map[fname]
    
    return pr_data_copy, truncated_count


def extract_content_from_response(resp: Any) -> str:
    try:
        return resp.choices[0].message.content
    except Exception:
        try:
            return resp.choices[0]["message"]["content"]
        except Exception:
            try:
                return resp.choices[0].text
            except Exception:
                return str(resp)


def process_batch(
    tasks: List[Tuple[str, str]], # (pr_id, variation)
    all_pr_data: Dict[str, Any],
    client: Any, 
    builder_cls: Any,
    args: argparse.Namespace, 
    exp_id: str,
    out_path: str,
    router_type: str,
    tokenizer: Any = None
):
    """
    Process a batch of tasks.
    """
    results = []
    
    for pr_id, variation in tasks:
        pr_data = all_pr_data[pr_id]
        
        messages, truncated_count = build_messages_for(
            builder_cls, pr_data, variation,
            truncate_p4=args.truncate_p4,
            tokenizer=tokenizer,
            max_tokens=args.max_context_tokens
        )
        
        # Extract prompt text for logging/hashing
        prompt_text = "\n\n".join([m["content"] for m in messages])
        prompt_hash = hashlib.md5(prompt_text.encode("utf-8")).hexdigest()

        content = ""
        input_tokens = 0
        output_tokens = 0
        finish_reason = "unknown"
        provider_used = args.provider if args.provider else router_type
        
        if not args.dry_run:
            try:
                call_kwargs = {
                    "model": args.model,
                    "messages": messages,
                    "temperature": args.temperature,
                }

                if args.max_tokens is not None:
                    call_kwargs["max_tokens"] = args.max_tokens
                
                # OpenRouter-specific provider configuration
                if router_type == "openrouter" and (args.provider or args.quantization):
                    extra_body = {}
                    if args.provider:
                        extra_body["provider"] = {
                            "order": [args.provider],
                            "allow_fallbacks": False
                        }
                    if args.quantization:
                        if "provider" not in extra_body: extra_body["provider"] = {}
                        extra_body["provider"]["quantizations"] = args.quantization
                    call_kwargs["extra_body"] = extra_body
                
                elif router_type == "huggingface" and args.provider:
                     if ":" not in call_kwargs["model"]:
                        call_kwargs["model"] = f"{call_kwargs['model']}:{args.provider}"

                resp = client.chat.completions.create(**call_kwargs)
                
                content = extract_content_from_response(resp)
                
                # Usage extraction
                try:
                    if hasattr(resp, "usage") and resp.usage:
                        input_tokens = resp.usage.prompt_tokens
                        output_tokens = resp.usage.completion_tokens
                    if hasattr(resp, "choices") and resp.choices:
                        finish_reason = resp.choices[0].finish_reason
                except Exception:
                    pass

                # Clean content
                try:
                    clean_content = content.strip()
                    if clean_content.startswith("```json"):
                        clean_content = clean_content[7:]
                    elif clean_content.startswith("```"):
                        clean_content = clean_content[3:]
                    if clean_content.endswith("```"):
                        clean_content = clean_content[:-3]
                    parsed = json.loads(clean_content.strip())
                    if isinstance(parsed, dict) and "description" in parsed:
                        content = parsed["description"]
                except Exception:
                    pass
                
            except Exception as e:
                # Log error but don't crash worker
                error_msg = f"Error: {e}"
                content = json.dumps({"description": "ERROR_GENERATION_FAILED", "error": str(e)})
        else:
            # Dry run
            # logger.info(f"Dry Run {pr_id}/{variation}")
            pass

        # Decoding params representation
        _decoding = {"temperature": args.temperature}
        if args.max_tokens is not None:
            _decoding["max_tokens"] = args.max_tokens
        decoding_params = json.dumps(_decoding)

        record = ExperimentRecord(
            experiment_id=exp_id,
            timestamp=datetime.now().isoformat(),
            model=args.model,
            pr_id=pr_id,
            variation=variation,
            seed=args.seed,
            decoding_params=decoding_params,
            prompt_hash=prompt_hash,
            prompt_token_count=input_tokens,
            context_token_count=input_tokens,
            generated_token_count=output_tokens,
            truncated_token_count=truncated_count,
            full_prompt=prompt_text,
            generated_text=content,
            device_map="api",
            finish_reason=str(finish_reason),
            provider=provider_used,
            precision=" ".join(args.quantization) if args.quantization else "auto"
        )

        results.append(asdict(record))

    if results:
        df = pd.DataFrame(results)
        if os.path.exists(out_path):
            df.to_csv(out_path, mode='a', header=False, index=False)
        else:
            df.to_csv(out_path, index=False)

def run_worker(
    worker_id: int,
    tasks: List[Tuple[str, str]],
    all_pr_data: Dict[str, Any],
    builder_cls: Any,
    args: argparse.Namespace,
    exp_id: str,
    base_out_path: str,
    router_type: str
):
    """Worker process."""
    log_file = os.path.join(LOG_DIR, f"openai_worker_{worker_id}.log")
    w_logger = setup_logger(f"worker_{worker_id}", log_file)
    w_logger.info(f"Worker {worker_id} started with {len(tasks)} tasks")
    
    # Construct shard path
    base_name, ext = os.path.splitext(base_out_path)
    shard_path = f"{base_name}_shard_{worker_id}{ext}"
    
    # Initialize client
    client = None
    if not args.dry_run:
        client = create_client(args.base_url, api_key=args.api_key)

    # Initialize tokenizer if needed
    tokenizer = None
    if args.truncate_p4:
         hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
         tokenizer = get_tokenizer_for_model(args.model, hf_token=hf_token)

    batch_size = 5
    total_batches = (len(tasks) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        batch = tasks[i*batch_size : (i+1)*batch_size]
        w_logger.info(f"Processing batch {i+1}/{total_batches}")
        
        process_batch(
            batch,
            all_pr_data,
            client,
            builder_cls,
            args,
            exp_id,
            shard_path,
            router_type,
            tokenizer
        )
        
    w_logger.info("Worker finished.")

def merge_shards(main_results_path: str, cleanup: bool = True):
    base_name, ext = os.path.splitext(main_results_path)
    pattern = f"{base_name}_shard_*{ext}"
    shard_files = glob.glob(pattern)
    
    if not shard_files:
        return

    print(f"Merging {len(shard_files)} shard files...")
    
    dfs = []
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
        if 'pr_id' in full_df.columns and 'variation' in full_df.columns:
            full_df = full_df.drop_duplicates(subset=['pr_id', 'variation'], keep='last')
        
        full_df.to_csv(main_results_path, index=False)
        print(f"Merged results saved to {main_results_path}")
        
        if cleanup:
            for f in shard_files:
                os.remove(f)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="Optional API key")
    parser.add_argument("--model", default="meta-llama-3.1-8b-instruct", help="Model name")
    parser.add_argument("--builder", default="auto", choices=["auto", "llama", "qwen"])
    parser.add_argument("--limit", type=int, default=0, help="Number of PRs to limit to (0 for all)")
    parser.add_argument("--variations", nargs="+", default=["P-6"], choices=list(CONTEXT_MAPPING.keys())+["ALL"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=DECODING_PARAMS['temperature'])
    parser.add_argument("--provider", default=None)
    parser.add_argument("--quantization", nargs="+", default=None)
    parser.add_argument("--router", default="openrouter", choices=["openrouter", "huggingface"])
    parser.add_argument("--truncate-p4", action="store_true")
    parser.add_argument("--max-context-tokens", type=int, default=30000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    if "ALL" in args.variations:
        variations = list(CONTEXT_MAPPING.keys())
    else:
        variations = args.variations

    # Router & API Key Logic
    router_type = args.router
    if "huggingface.co" in args.base_url:
        router_type = "huggingface"

    if not args.api_key and router_type == "huggingface":
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            args.api_key = hf_token

    # Prompt Builder
    builder_cls = select_prompt_builder(args.builder, args.model)
    logger.info(f"Builder: {builder_cls.__name__}")

    # Load Data
    loader = DataLoader()
    all_pr_data = loader.get_comprehensive_data()
    
    if args.limit > 0:
        keys = list(all_pr_data.keys())[:args.limit]
        all_pr_data = {k: all_pr_data[k] for k in keys}
        
    logger.info(f"Loaded {len(all_pr_data)} PRs")

    # Output Path & Exp ID
    ts = int(time.time())
    model_slug = args.model.replace("/", "_").replace(":", "-")
    exp_id = f"exp_{model_slug}_{ts}"
    out_path = args.out or os.path.join(LOG_DIR, f"{model_slug}_{ts}.csv")

    if args.resume:
        out_path = args.resume
        # Try to read old exp_id
        if os.path.exists(out_path):
             try:
                 df = pd.read_csv(out_path)
                 if 'experiment_id' in df.columns: exp_id = str(df.iloc[0]['experiment_id'])
             except: pass

    # Recovery
    merge_shards(out_path, cleanup=True)
    
    # Deduplicate work
    completed_pairs = set()
    if os.path.exists(out_path):
        try:
            df = pd.read_csv(out_path)
            for _, row in df.iterrows():
                # Check for failure markers
                gen_text = str(row.get('generated_text', ''))
                finish_reason = str(row.get('finish_reason', ''))
                
                # We skip marking as complete if it's an error or length truncation (loops)
                if "ERROR_GENERATION_FAILED" in gen_text:
                    continue
                if finish_reason == "length":
                     continue
                     
                completed_pairs.add((str(row['pr_id']), row['variation']))
            print(f"Found {len(completed_pairs)} completed tasks.")
        except: pass
        
    # Generate Task List
    all_tasks = []
    for pr_id in all_pr_data:
        for var in variations:
            if (str(pr_id), var) not in completed_pairs:
                all_tasks.append((pr_id, var))
                
    print(f"To process: {len(all_tasks)} tasks")

    if args.workers > 1:
        print(f"Spawning {args.workers} workers...")
        chunk_size = math.ceil(len(all_tasks) / args.workers)
        chunks = [all_tasks[i:i + chunk_size] for i in range(0, len(all_tasks), chunk_size)]
        
        processes = []
        for i, chunk in enumerate(chunks):
            p = multiprocessing.Process(
                target=run_worker,
                args=(i, chunk, all_pr_data, builder_cls, args, exp_id, out_path, router_type)
            )
            processes.append(p)
            p.start()
            
        for p in processes:
            p.join()
            
        merge_shards(out_path)
        
    else:
        # Sequential
        client = create_client(args.base_url, api_key=args.api_key) if not args.dry_run else None
        tokenizer = None
        if args.truncate_p4:
            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            tokenizer = get_tokenizer_for_model(args.model, hf_token=hf_token)

        batch_size = 10
        total_batches = (len(all_tasks) + batch_size - 1) // batch_size
        
        for i in range(total_batches):
            batch = all_tasks[i*batch_size : (i+1)*batch_size]
            print(f"Processing batch {i+1}/{total_batches}")
            process_batch(
                batch,
                all_pr_data,
                client,
                builder_cls,
                args,
                exp_id,
                out_path,
                router_type,
                tokenizer
            )
            
    cleanup_duplicates(out_path)
    print("Done.")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
