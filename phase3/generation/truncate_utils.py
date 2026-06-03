"""
Intelligent Truncation Strategy for Generation Phase.

When generating prompts for large PRs (especially using P-4 or full context),
file contents often exceed the token limit of the target model. This module
implements a two-stage truncation strategy to fit content within a budget
while preserving the most critical information.

## Strategy Outline

1. **Global Allocation (Proportional Sizing)**:
   - Calculate the total token count of all file contents in the PR.
   - If the total exceeds the budget (`max_total_tokens`), apply a global scaling factor:
       `scale = max_total_tokens / total_tokens`
   - Allocate a token budget to each file proportional to its original size:
       `file_budget = original_tokens * scale`
   - Ensure a minimum token count (e.g., 100 tokens) per file to avoid empty files.

2. **Per-File Truncation (Head & Tail Preservation)**:
   - For each file exceeding its allocated budget, remove content from the *middle*.
   - Preserve both the beginning (e.g., imports, class definitions) and the end
     (often containing critical logic or tests) of the file.
   - Insert a placeholder marker indicating the number of truncated lines.
   - Typically preserves 40% head and 40% tail of the *lines* that fit within the
     allocated token budget, discarding the middle 20%.

3. **Tokenizer Awareness**:
   - Uses model-specific tokenizers (e.g., `tiktoken` for OpenAI, `transformers` for
     Hugging Face models) to ensure accurate counts.
   - Accounts for overhead like file headers in the final prompt structure.

## Usage

    truncated_files = truncate_pr_files(
        files=pr_data['files'],
        max_total_tokens=30000,
        tokenizer=tokenizer
    )
"""
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_tokens(text: str, tokenizer) -> int:
    """
    Estimate token count using the provided tokenizer.
    
    Perf Opt: If text is extremely large (>500k chars), use a heuristic
    to avoid tokenizer hangs/slowdowns.
    """
    if not text:
        return 0
        
    # Heuristic for massive files (avoid tiktoken hang on >500k chars)
    if len(text) > 500_000:
        return len(text) // 4
        
    try:
        if hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
        # Fallback for some HF tokenizers that might not have direct encode
        # or if it's a different type of object
        return len(text) // 4
    except Exception:
        # Fallback if encoding fails
        return len(text) // 4


def truncate_file_content(
    content: str,
    max_tokens: int,
    tokenizer,
    preserve_ratio: float = 0.4
) -> str:
    """
    Truncate a single file's content to fit within max_tokens.
    
    Strategy:
    - Keep the beginning (preserve_ratio) and end (preserve_ratio) of the file
    - Remove middle content
    - Insert a marker where content is removed
    
    Args:
        content: The file content to truncate
        max_tokens: Maximum tokens allowed for this file
        tokenizer: Tokenizer to use for counting
        preserve_ratio: Ratio of content to keep from beginning and end (default 0.4 = 40% each)
    
    Returns:
        Truncated content string
    """
    current_tokens = estimate_tokens(content, tokenizer)
    
    if current_tokens <= max_tokens:
        return content
    
    # Split by lines for better truncation
    lines = content.split('\n')
    total_lines = len(lines)
    
    # Handle files with few lines (e.g., minified code) or just short files
    if total_lines <= 10:
        # Optimization: If content is massive (e.g. minified JS > 1MB), 
        # truncate string first to avoid tokenizer hang.
        # 1M chars is roughly 250k tokens. If max_tokens is e.g. 30k, 
        # we can safely cut at 200k chars (approx 50k tokens) and then refine.
        if len(content) > max_tokens * 10:
            logger.warning(f"File content extremely large ({len(content)} chars) with few lines. Pre-truncating.")
            content = content[:max_tokens * 5] # Hard cut to safe length
            
        tokens = tokenizer.encode(content)
        if len(tokens) > max_tokens:
            truncated_tokens = tokens[:max_tokens - 10]  # Leave room for marker
            return tokenizer.decode(truncated_tokens) + "\n\n[... CONTENT TRUNCATED ...]\n"
        return content
    
    # Calculate how many lines to keep from beginning and end
    keep_lines_start = int(total_lines * preserve_ratio)
    keep_lines_end = int(total_lines * preserve_ratio)
    
    # Binary search for the right number of lines to keep
    low = 0
    # max possible lines we could theoretically keep (upper bound)
    high = keep_lines_start + keep_lines_end 
    
    best_truncated = ""
    
    # We want to find the maximum `k` such that keeping `k` lines 
    # (distributed between start/end) fits within budget.
    
    # Simple binary search implementation
    # We will search for 'lines_to_keep' in range [0, high]
    
    while low <= high:
        mid = (low + high) // 2
        
        # Distribute 'mid' lines between start and end according to ratio
        # If mid is total lines to keep, how many at start?
        # Ratio was 50/50 typically (preserve_ratio was applied to total originally)
        # But here preserve_ratio is 0.4 start, 0.4 end. So equal weight.
        
        if mid == 0:
            current_keep_start = 0
            current_keep_end = 0
        else:
            current_keep_start = mid // 2
            current_keep_end = mid - current_keep_start
            
        # Ensure we don't exceed original file
        if current_keep_start + current_keep_end >= total_lines:
             current_keep_start = keep_lines_start # fallback to safe max
             current_keep_end = keep_lines_end
             
        beginning = '\n'.join(lines[:current_keep_start])
        end = '\n'.join(lines[-current_keep_end:]) if current_keep_end > 0 else ""
        
        marker = f"\n\n[... TRUNCATED {total_lines - current_keep_start - current_keep_end} LINES ...]\n\n"
        truncated = beginning + marker + end
        
        tokens = estimate_tokens(truncated, tokenizer)
        
        if tokens <= max_tokens:
            best_truncated = truncated
            low = mid + 1 # Try to keep more
        else:
            high = mid - 1 # Keep fewer
            
    if best_truncated:
        return best_truncated
    
    # Fallback: just keep first N lines
    marker = f"\n\n[... TRUNCATED {total_lines - 10} LINES ...]\n\n"
    return '\n'.join(lines[:10]) + marker


def truncate_pr_files(
    files: List[Dict[str, Any]],
    max_total_tokens: int,
    tokenizer,
    preserve_ratio: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Truncate multiple PR files to fit within a total token budget.
    
    Strategy:
    1. Calculate total tokens needed
    2. If under budget, return unchanged
    3. Otherwise, distribute budget evenly across files
    4. Truncate each file to its allocation
    
    Args:
        files: List of file dicts with 'path' and 'content' keys
        max_total_tokens: Maximum tokens for all files combined
        tokenizer: Tokenizer to use for counting
        preserve_ratio: Ratio to preserve from beginning/end of each file
    
    Returns:
        List of file dicts with potentially truncated content
    """
    if not files:
        return files
    
    # Calculate current token usage
    total_tokens = sum(estimate_tokens(f.get('content', ''), tokenizer) for f in files)
    
    if total_tokens <= max_total_tokens:
        logger.info(f"Files fit within budget: {total_tokens}/{max_total_tokens} tokens")
        return files
    
    logger.warning(f"Files exceed budget: {total_tokens}/{max_total_tokens} tokens. Truncating proportionally...")
    
    # Calculate file sizes to distribute budget proportionally
    file_token_counts = []
    for f in files:
        count = estimate_tokens(f.get('content', ''), tokenizer)
        file_token_counts.append(count)
    
    # Reserve tokens for headers (approximate overhead) across all files
    # Only if we assumed headers were not part of the content being estimated, 
    # but here 'content' is the raw file content. 
    # The prompt builder adds headers, but here we are truncating the raw content.
    # We need to account for the overhead we add per file in the final prompt?
    # The previous code reserved tokens for header per file within the allocated budget.
    
    truncated_files = []
    
    for i, file_info in enumerate(files):
        path = file_info.get('path', 'unknown')
        content = file_info.get('content', '')
        original_tokens = file_token_counts[i]
        
        # Calculate proportional budget
        # Formula: (FileTokens / TotalTokens) * MaxTokens
        # We also need to subtract overhead for headers if we strictly follow the budget.
        # Let's say overhead is ~20 tokens per file.
        # adjusted_max = max_total_tokens - (num_files * 20)
        # For simplicity, we stick to the user's request of proportional division of the passed limit.
        
        if total_tokens > 0:
            proportion = original_tokens / total_tokens
            tokens_for_this_file = int(max_total_tokens * proportion)
        else:
            tokens_for_this_file = 0

        # Reserve some tokens for file header/footer wrapper that will be added later or during estimation
        header = f"File: {path}\n```\n"
        footer = "\n```"
        header_tokens = estimate_tokens(header + footer, tokenizer)
        
        # Determine actual content budget
        available_for_content = max(100, tokens_for_this_file - header_tokens)
        
        # Log decision
        logger.debug(f"File {path}: {original_tokens} -> {tokens_for_this_file} tokens (alloc)")

        truncated_content = truncate_file_content(
            content,
            available_for_content,
            tokenizer,
            preserve_ratio
        )
        
        truncated_files.append({
            'path': path,
            'content': truncated_content,
            'truncated': truncated_content != content
        })
    
    # Verify total
    final_total = sum(estimate_tokens(f.get('content', ''), tokenizer) for f in truncated_files)
    logger.info(f"After truncation: {final_total}/{max_total_tokens} tokens ({len(truncated_files)} files)")
    
    return truncated_files


def get_tokenizer_for_model(model_name: str, hf_token: Optional[str] = None):
    """
    Load the appropriate tokenizer for a given model.
    
    Args:
        model_name: Model identifier (e.g., "Qwen/Qwen2.5-72B-Instruct")
        hf_token: Optional HuggingFace token for gated models (like Llama 3)
    
    Returns:
        Loaded tokenizer
    """
    # Handle OpenAI models with tiktoken
    if "gpt" in model_name.lower():
        try:
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model(model_name)
                logger.info(f"Loading tiktoken encoder for: {model_name}")
                return enc
            except KeyError:
                logger.info(f"Loading tiktoken encoder for fallback: gpt-4o")
                return tiktoken.encoding_for_model("gpt-4o")
        except ImportError:
            logger.warning("tiktoken not found, falling back to transformers for gpt model")

    from transformers import AutoTokenizer
    
    # Map model names to tokenizer identifiers
    if "qwen" in model_name.lower():
        tokenizer_name = "Qwen/Qwen2.5-72B-Instruct"
    elif "llama" in model_name.lower():
        tokenizer_name = "meta-llama/Llama-3.1-70B-Instruct"
    else:
        # Default to a common tokenizer
        tokenizer_name = model_name
    
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    return AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
