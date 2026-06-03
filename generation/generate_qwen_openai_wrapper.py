"""
Thin wrapper that forwards to `generate_openai.py` with the Qwen prompt builder.
"""
import sys
import os

def main():
    # Default builder/model for Qwen
    default_args = [
        "--builder", "qwen",
        "--model", "Qwen/Qwen2.5-72B-Instruct"
    ]

    # Preserve CLI passthrough (user can override defaults)
    new_argv = [sys.argv[0]] + default_args + sys.argv[1:]
    sys.argv = new_argv

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from generation.generate_openai import main as unified_main
    unified_main()


if __name__ == "__main__":
    main()
