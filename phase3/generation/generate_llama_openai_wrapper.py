"""
Thin wrapper that forwards to `generate_openai.py` with the Llama prompt builder.
"""
import sys
import os

def main():
    # Default builder/model for Llama
    default_args = [
        "--builder", "llama",
        "--model", "meta-llama-3.1-8b-instruct"
    ]

    # Preserve CLI passthrough (user can override defaults)
    new_argv = [sys.argv[0]] + default_args + sys.argv[1:]
    sys.argv = new_argv

    # Ensure project root is importable (same pattern as other scripts)
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from phase3.generation.generate_openai import main as unified_main
    unified_main()


if __name__ == "__main__":
    main()
