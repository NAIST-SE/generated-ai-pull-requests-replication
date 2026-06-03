# P-4 Truncation Usage Guide

## Problem
P-4 (File Contents) variation can exceed the 32K token limit on providers like Hyperbolic and Together AI, causing generation failures.

## Solution
Intelligent file content truncation that preserves the most important parts of each file while fitting within the token budget.

## Usage

### Enable Truncation for P-4

```bash
pyenv exec python phase3/generation/generate_openai.py \
  --base-url "https://router.huggingface.co/v1" \
  --api-key "$HF_TOKEN" \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --builder qwen \
  --router huggingface \
  --provider hyperbolic \
  --truncate-p4 \
  --max-context-tokens 30000 \
  --limit 10 \
  --variations P-4
```

### Key Flags

- `--truncate-p4`: Enable intelligent truncation for P-4 variation
- `--max-context-tokens 30000`: Maximum tokens for file contents (default: 30000)
  - Leaves room for 512 output tokens + system overhead
  - Total should stay under 32,768 token limit

### How It Works

1. **Token Budget Distribution**: Divides `max-context-tokens` evenly across all files
2. **Smart Truncation Strategy**:
   - Keeps **beginning** of each file (40% by default) - contains imports, headers, function signatures
   - Keeps **end** of each file (40% by default) - contains return statements, closing logic
   - Removes **middle** content if truncation is needed
   - Inserts marker: `[... TRUNCATED N LINES ...]`
3. **Preserves Structure**: File paths, code structure, and key logic remain intact

### Example Output

```python
File: src/main.py
```
import os
import sys
from typing import List

def main():
    """Main entry point."""
    config = load_config()

[... TRUNCATED 150 LINES ...]

    return results

if __name__ == "__main__":
    main()
```
```

### Adjusting Token Budget

For different models or providers:

```bash
# For models with 64K context (e.g., DeepSeek)
--max-context-tokens 60000

# For models with 128K context (e.g., Claude)
--max-context-tokens 120000

# Conservative for 32K limit
--max-context-tokens 28000
```

### Without Truncation (Default)

If you don't use `--truncate-p4`, the script will attempt to send full file contents and may fail for large PRs:

```bash
# This may fail on 5/10 PRs with 32K limit
pyenv exec python phase3/generation/generate_openai.py \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --variations P-4
```

## Testing Truncation

Use the notebook to preview which PRs need truncation:

```bash
# Open the notebook
jupyter notebook phase3/analyze_p4_tokens.ipynb
```

This will show:
- Which PRs exceed 32K limit
- How many tokens each PR needs
- Token distribution across all PRs

## Advanced: Custom Truncation Ratio

Edit `truncate_utils.py` to change how much content is preserved from beginning/end:

```python
truncated_files = truncate_pr_files(
    files_for_truncation,
    max_total_tokens=max_tokens,
    tokenizer=tokenizer,
    preserve_ratio=0.5  # Keep 50% from start, 50% from end (instead of 40/40)
)
```
