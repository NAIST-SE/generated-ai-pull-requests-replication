"""
Controlled Llama-3.1-70B-Instruct Generation Experiment (Phase 3)

This script implements rigorous, reproducible text generation for PR descriptions
using the Llama-3.1-70B-Instruct model from Hugging Face Transformers.

It adheres to strict experimental controls:
- Fixed decoding parameters
- Deterministic seeding
- Comprehensive logging
- Systematic context variation
"""

import os
import sys
import json
import time
import random
import re
import logging
import argparse
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
import transformers
from packaging import version

# --- Dependency Check ---
def check_dependencies():
    """Verifies that installed libraries support Llama 3.1 and 4-bit quantization."""
    req_transformers = "4.43.0"
    if version.parse(transformers.__version__) < version.parse(req_transformers):
        raise ImportError(
            f"Llama 3.1 requires transformers >= {req_transformers}, but you have {transformers.__version__}. "
            "Please upgrade: pip install --upgrade transformers accelerate bitsandbytes"
        )
check_dependencies()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import DataLoader, CONTEXT_MAPPING
from generation.config import DECODING_PARAMS

# --- Configuration & Controlled Variables ---

MODEL_ID = "meta-llama/Llama-3.1-70B-Instruct"

# Fixed Decoding Parameters (Loaded from config.py)
# DECODING_PARAMS imported above

# Context Window Limit (Llama 3.1 70B supports 128k)
# NO TRUNCATION - Full context for experimental parity with ChatGPT
# MAX_INPUT_TOKENS = 110000  
# TRUNCATION_STRATEGY = "tail"

# Logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "../datasets")
LOG_FILE = os.path.join(LOG_DIR, "llama_generation.log")
# Module-level logger; do NOT configure handlers at import time. Call
# `configure_logging()` from the entrypoint when running as a script.
logger = logging.getLogger("LlamaExperiment")


def configure_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure logging handlers for this module.

    This avoids configuring file handlers at import time (which would
    affect other scripts that merely import `PromptBuilder`).
    """
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

# --- Data Structures ---

@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    model: str
    pr_id: str
    variation: str
    seed: int
    decoding_params: str  # JSON string
    prompt_hash: str
    prompt_token_count: int
    context_token_count: int
    generated_token_count: int
    full_prompt: str
    generated_text: str
    device_map: str = "unknown" # Added for External Validity (Hardware Sharding)
    finish_reason: str = "unknown"

# --- Prompt Construction ---

class PromptBuilder:
    """Builds prompts with strict adherence to variations."""

    SYSTEM_PROMPT = (
        "You are a helpful assistant that generates GitHub pull request descriptions. "
        "Based on the provided context, write a clear and informative description. "
        "Respond with ONLY a JSON object in this format:\n"
        "{\"description\": \"<your description here>\"}\n"
        "Do not include any other text or explanation."
    )

    @staticmethod
    def construct_context_block(pr_data: Dict[str, Any], variation: str, tokenizer=None) -> str:
        """
        Constructs the variable context block based on the variation ID.
        
        Args:
            pr_data: PR data dictionary
            variation: Context variation (P-1 through P-6)
            tokenizer: Optional tokenizer for precise token counting (recommended)
        """
        import math
        
        def safe_str(value, default=''):
            """Safely convert value to string, handling NaN and None."""
            if value is None:
                return default
            if isinstance(value, float) and math.isnan(value):
                return default
            return str(value)
        
        parts = []
        files = pr_data['files']
        info = pr_data['info']
        issues = pr_data['issues']

        # P-1: Repo & File Paths
        if variation in ["P-1", "P-6"]:
            repo_name = safe_str(info.get('repository_name_with_owner'), 'Unknown')
            path_list = [safe_str(f.get('filename')) for f in files if safe_str(f.get('filename'))]
            parts.append(f"<repository>{repo_name}</repository>")
            if path_list:
                parts.append("<affected_files>\n" + "\n".join(path_list) + "\n</affected_files>")

        # P-2: Title
        if variation in ["P-2", "P-6"]:
            title = safe_str(info.get('title'))
            if title:
                parts.append(f"<title>{title}</title>")

        # P-3: Diffs
        if variation in ["P-3", "P-6"]:
            diff_text = []
            for f in files:
                fname = safe_str(f.get('filename'))
                patch = safe_str(f.get('patch'))
                if patch and fname:
                    # Include full patch for per-file diffs (global truncation still applies later)
                    diff_text.append(f"File: {fname}\n```diff\n{patch}\n```")
            if diff_text:
                parts.append("<diffs>\n" + "\n\n".join(diff_text) + "\n</diffs>")

        # P-4: File Contents
        if variation == "P-4":
            # Simple allocation strategy:
            # 1. Reserve 2k tokens for system prompt overhead (~200-500 actual)
            # 2. If total fits in 108k tokens → use all files completely (97.9% of PRs)
            # 3. If exceeds budget → proportional truncation with descending size order (2.1% of PRs)
            TOTAL_TOKEN_BUDGET = 110000  # Conservative limit to prevent OOM
            RESERVED_TOKENS = 2000
            AVAILABLE_TOKENS = TOTAL_TOKEN_BUDGET - RESERVED_TOKENS  # 108k tokens
            
            # Count files with content
            files_with_content = [f for f in files if safe_str(f.get('file_content'))]
            
            if len(files_with_content) > 0:
                # Calculate total tokens using actual tokenizer if available, else estimate
                if tokenizer is not None:
                    # Precise token counting with actual tokenizer
                    file_token_counts = []
                    for f in files_with_content:
                        content = safe_str(f.get('file_content'))
                        if content:
                            tokens = len(tokenizer.encode(content, add_special_tokens=False))
                            file_token_counts.append((f, tokens))
                        else:
                            file_token_counts.append((f, 0))
                    
                    total_tokens = sum(count for _, count in file_token_counts)
                else:
                    # Fallback: rough character-based estimate (4 chars ≈ 1 token)
                    CHARS_PER_TOKEN = 4
                    file_token_counts = []
                    for f in files_with_content:
                        content = safe_str(f.get('file_content'))
                        if content:
                            estimated_tokens = len(content) // CHARS_PER_TOKEN
                            file_token_counts.append((f, estimated_tokens))
                        else:
                            file_token_counts.append((f, 0))
                    
                    total_tokens = sum(count for _, count in file_token_counts)
                
                content_text = []
                
                if total_tokens <= AVAILABLE_TOKENS:
                    # All files fit - use full content (97.9% of PRs)
                    for f, _ in file_token_counts:
                        fname = safe_str(f.get('filename'))
                        content = safe_str(f.get('file_content'))
                        if content and fname:
                            content_text.append(f"File: {fname}\n```\n{content}\n```")
                else:
                    # Exceeds budget - proportional truncation with size-based sorting (2.1% of PRs)
                    # Sort files by token count (descending) - larger files processed first
                    sorted_file_tokens = sorted(file_token_counts, key=lambda x: x[1], reverse=True)
                    
                    # Calculate truncation ratio to fit within budget
                    truncation_ratio = AVAILABLE_TOKENS / total_tokens
                    MIN_TOKENS = 250  # Minimum per file (~1000 chars)
                    
                    for f, original_tokens in sorted_file_tokens:
                        fname = safe_str(f.get('filename'))
                        content = safe_str(f.get('file_content'))
                        
                        if content and fname:
                            allocated_tokens = int(original_tokens * truncation_ratio)
                            allocated_tokens = max(allocated_tokens, MIN_TOKENS)
                            
                            # Truncate content to allocated token budget
                            if tokenizer is not None:
                                # Precise: encode, truncate tokens, decode
                                tokens = tokenizer.encode(content, add_special_tokens=False)
                                truncated_tokens = tokens[:allocated_tokens]
                                content_snippet = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                            else:
                                # Fallback: character-based approximation
                                allocated_chars = allocated_tokens * 4
                                content_snippet = content[:allocated_chars]
                            
                            if original_tokens > allocated_tokens:
                                content_snippet += "\n... (truncated)"
                            
                            content_text.append(f"File: {fname}\n```\n{content_snippet}\n```")
                
                if content_text:
                    parts.append("<files>\n" + "\n\n".join(content_text) + "\n</files>")

        # P-5: Issues
        if variation in ["P-5", "P-6"]:
            issue_texts = []
            for issue in issues:
                title = safe_str(issue.get('title')) or safe_str(issue.get('issue_title'))
                body = safe_str(issue.get('body')) or safe_str(issue.get('issue_body'))
                if title or body:
                    issue_texts.append(f"Title: {title}\nBody: {body}")
            
            # Legacy fallback
            if not issue_texts:
                 i_titles = safe_str(info.get('issue_titles'))
                 i_bodies = safe_str(info.get('issue_bodies'))
                 if i_titles:
                     issue_texts.append(f"Titles: {i_titles}\nBodies: {i_bodies}")

            if issue_texts:
                parts.append("<issues>\n" + "\n---\n".join(issue_texts) + "\n</issues>")

        return "\n\n".join(parts)

    @staticmethod
    def wrap_context_in_prompt(context_block: str) -> List[Dict[str, str]]:
        """Wraps the (potentially truncated) context block in the chat template."""
        user_content = (
            "### CONTEXT\n"
            f"{context_block}\n"
            "### END CONTEXT\n\n"
            "Reminder: based on the provided context, write a clear and informative description. "
            "Respond with ONLY a JSON object in this format: {\"description\": \"...\"}"
        )

        return [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

    # Deprecated: use run_experiment logic to separate construction and wrapping
    @staticmethod
    def build_chat_messages(pr_data: Dict[str, Any], variation: str, tokenizer=None) -> List[Dict[str, str]]:
        context = PromptBuilder.construct_context_block(pr_data, variation, tokenizer)
        return PromptBuilder.wrap_context_in_prompt(context)

# --- Experiment Execution ---

def setup_device():
    if torch.cuda.is_available():
        return "cuda" # Accelerate will handle Map
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(device, quantization: str = "4bit", cpu_offload: bool = False, max_memory: Optional[Dict[str, str]] = None):
    """
    Load model with specified quantization strategy.
    
    Args:
        device: Target device (cuda/mps/cpu)
        quantization: Quantization mode - '4bit', '8bit', or 'none'
        cpu_offload: Enable CPU offloading for 8-bit (spillover to system RAM)
        max_memory: Optional dict mapping device IDs to max memory (e.g., {0: "38GiB", "cpu": "100GiB"})
    """
    logger.info(f"Loading model {MODEL_ID} with {quantization} quantization...")
    try:
        # Read HF token from env (keeps credentials out of source)
        hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if hf_token:
            logger.info("Hugging Face token found in environment; using for authenticated model access.")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        else:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token 
        
        # Threat Mitigation: Construct Validity (Numerical Stability)
        # Use bfloat16 for Llama-3.1-70B to avoid underflow in long contexts.
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        # Configure quantization based on user selection
        bnb_config = None
        quant_description = "none (full precision)"
        
        if quantization == "4bit":
            # 4-bit quantization: ~18-24GB VRAM for 70B model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            quant_description = "4-bit NF4"
        elif quantization == "8bit":
            # 8-bit quantization: ~35-40GB VRAM for 70B model
            # With CPU offload: can use system RAM for overflow
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=cpu_offload,  # Allow spillover to RAM
            )
            quant_description = f"8-bit{' (CPU offload)' if cpu_offload else ''}"
            if cpu_offload:
                logger.info("CPU offloading enabled - model weights can spill to system RAM")
        elif quantization == "none":
            # Full precision: ~140GB VRAM for 70B model
            quant_description = f"full precision ({dtype})"
        else:
            raise ValueError(f"Invalid quantization mode: {quantization}. Choose '4bit', '8bit', or 'none'.")

        # Detect FlashAttention2 availability at runtime
        attn_impl = "sdpa"  # Force SDPA instead of eager (which OOMs) if FA2 is missing
        try:
            import flash_attn  # type: ignore
            if torch.cuda.is_available():
                attn_impl = "flash_attention_2"
        except Exception:
            logger.info("FlashAttention2 not available; using sdpa implementation.")

        # Build model loading configuration
        load_kwargs = {
            "device_map": "auto",  # Critical for 70B multi-GPU sharding
            "trust_remote_code": False,
            "low_cpu_mem_usage": True,  # Required for quantized models with device_map
        }
        
        # Add max_memory if specified (caps GPU usage, enables CPU offload)
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
            logger.info(f"Max memory limits: {max_memory}")
        
        # Add quantization config if using quantization
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
        else:
            # Use dtype for full precision loading
            load_kwargs["torch_dtype"] = dtype
            
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl

        # If we have a token, pass it to from_pretrained so gated repos can be accessed
        if hf_token:
            load_kwargs["token"] = hf_token

        logger.info(f"Loading with {quant_description} and attn_impl={attn_impl}")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
        
        # Capture device map for External Validity
        device_map_str = str(getattr(model, "hf_device_map", "single_device"))
        logger.info(f"Model loaded with {quant_description} on map={device_map_str}")
        
        return model, tokenizer, device_map_str
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def run_experiment(
    model, 
    tokenizer, 
    device_map_str,
    pr_data: Dict[str, Any], 
    variation: str, 
    seed: int,
    experiment_id: str
) -> ExperimentRecord:
    
    # 1. Determinism
    set_seed(seed)
    
    # 2. Context Construction (Pass tokenizer for precise token counting)
    raw_context = PromptBuilder.construct_context_block(pr_data, variation, tokenizer)
    
    # Count context tokens for logging/recording only
    context_tokens = tokenizer(raw_context, add_special_tokens=False).input_ids
    context_tokens_count = len(context_tokens)
    
    # No truncation - use full context
    messages = PromptBuilder.wrap_context_in_prompt(raw_context)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 3. Final Tokenization (Safety Check)
    # For quantized models with device_map="auto", use first parameter's device
    # Move ONLY inputs, not the model (quantized models are pre-dispatched)
    input_device = next(model.parameters()).device
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]
    
    # No final safety truncation - allow full context through

    # 4. Generation
    start_time = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **DECODING_PARAMS,
                pad_token_id=tokenizer.eos_token_id
            )
    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("OOM detected. Clearing cache and retrying with aggressive cleanup.")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            # Don't raise - skip this PR and continue
            logger.warning(f"Skipping PR {pr_data['info']['id']} due to OOM")
            return None  # Signal to skip recording
        raise e
        
    duration = time.time() - start_time
    
    # 5. Decode
    # Slice off the input prompt length to get just the response
    generated_ids = outputs[0][len(inputs["input_ids"][0]):]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = len(generated_ids)
    
    # 6. Record
    return ExperimentRecord(
        experiment_id=experiment_id,
        timestamp=datetime.now().isoformat(),
        model=MODEL_ID,
        pr_id=pr_data['info']['id'],
        variation=variation,
        seed=seed,
        decoding_params=json.dumps(DECODING_PARAMS),
        prompt_hash=hashlib.md5(input_text.encode('utf-8')).hexdigest(),
        prompt_token_count=prompt_tokens,
        context_token_count=context_tokens_count,
        generated_token_count=generated_tokens,
        full_prompt=input_text, # Be careful with logging size
        generated_text=generated_text,
        device_map=device_map_str
    )

def main():
    parser = argparse.ArgumentParser(description="Run Llama-3.1-70B Experiment")
    parser.add_argument("--variations", nargs="+", default=["P-6"], choices=list(CONTEXT_MAPPING.keys()) + ["ALL"], help="Variations to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PRs for testing")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], 
                        help="Quantization mode: 4bit (~18-24GB), 8bit (~35-40GB), none (~140GB)")
    parser.add_argument("--cpu-offload", action="store_true", 
                        help="Enable CPU offloading for 8-bit (allows weights to spill to system RAM)")
    parser.add_argument("--max-memory-gpu", type=str, default=None,
                        help="Max memory per GPU (e.g., '38GiB'). Enables CPU offload when set.")
    parser.add_argument("--max-memory-cpu", type=str, default="100GiB",
                        help="Max memory for CPU offload (default: 100GiB)")
    parser.add_argument("--dry-run", action="store_true", help="Do not load model, just check data")
    
    args = parser.parse_args()
    
    # Expand ALL
    if "ALL" in args.variations:
        variations = list(CONTEXT_MAPPING.keys())
    else:
        variations = args.variations
    # Configure logging handlers now that we are running as a script
    configure_logging(LOG_FILE)
        
    logger.info("Initializing Experiment...")
    logger.info(f"Model: {MODEL_ID}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info(f"Variations: {variations}")
    logger.info(f"Seeds: {args.seeds}")
    
    # Load Data
    logger.info("Loading Data...")
    loader = DataLoader()
    all_pr_data = loader.get_comprehensive_data()
    
    prs_to_process = list(all_pr_data.items())
    if args.limit:
        prs_to_process = prs_to_process[:args.limit]
        
    logger.info(f"Loaded {len(prs_to_process)} PRs.")
    
    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return

    # Load Model
    device = setup_device()
    
    # Build max_memory config if GPU limit specified
    max_memory = None
    if args.max_memory_gpu:
        max_memory = {"cpu": args.max_memory_cpu}
        # Detect number of GPUs and apply limit to each
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                max_memory[i] = args.max_memory_gpu
            logger.info(f"Configured {num_gpus} GPUs with {args.max_memory_gpu} each, CPU: {args.max_memory_cpu}")
    
    model, tokenizer, device_map_str = load_model(
        device, 
        quantization=args.quantization, 
        cpu_offload=args.cpu_offload,
        max_memory=max_memory
    )
    
    # Prepare Output
    offload_suffix = "_cpuoffload" if args.cpu_offload else ""
    exp_id = f"exp_70b_{args.quantization}{offload_suffix}_{int(time.time())}"
    results_file = os.path.join(LOG_DIR, f"llama_70b_results_{exp_id}.csv")
    
    records = []
    
    # Execution Loop
    total_steps = len(prs_to_process) * len(variations) * len(args.seeds)
    current_step = 0
    
    logger.info(f"Starting execution loop: {total_steps} total generations.")
    
    for pr_id, pr_data in prs_to_process:
        for variation in variations:
            for seed in args.seeds:
                current_step += 1
                if current_step % 10 == 0:
                    logger.info(f"Progress: {current_step}/{total_steps}")
                
                try:
                    record = run_experiment(
                        model, tokenizer, device_map_str, pr_data, variation, seed, exp_id
                    )
                    
                    if record is None:  # OOM skip signal
                        continue
                        
                    records.append(asdict(record))
                    
                    # Periodic Save
                    if len(records) % 10 == 0:
                        pd.DataFrame(records).to_csv(results_file, index=False)
                    
                    # Aggressive memory cleanup after P-4 (heavy variation)
                    if variation == "P-4":
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    logger.error(f"Error processing PR {pr_id} / Var {variation}: {e}")
                    # Force GC on error
                    torch.cuda.empty_cache()
                    
    # Final Save
    if records:
        pd.DataFrame(records).to_csv(results_file, index=False)
        logger.info(f"Experiment completed. Results saved to {results_file}")
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    main()
