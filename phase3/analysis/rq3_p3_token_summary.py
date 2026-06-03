from pathlib import Path
from typing import Dict, List
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_loader import DataLoader

TOKEN_COLUMNS = [
    "prompt_token_count",
    "context_token_count",
    "generated_token_count",
    "truncated_token_count",
]


def normalize_model_name(raw: str) -> str:
    if not isinstance(raw, str):
        return "unknown"
    value = raw.strip()
    if not value:
        return "unknown"
    if "/" in value:
        value = value.split("/")[-1]

    lowered = value.lower()
    normalization = {
        "gpt-5-mini-2025-08-07": "gpt-5-mini-2025-08-07",
        "llama-3.1-8b-instruct": "Llama-3.1-8B-Instruct",
        "llama-3.1-70b-instruct": "Llama-3.1-70B-Instruct",
        "qwen2.5-72b-instruct": "Qwen2.5-72B-Instruct",
    }
    return normalization.get(lowered, value)


def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def summarize_tokens(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []

    for model_name, group in df.groupby("model_normalized"):
        row: Dict[str, object] = {"model": model_name, "N": len(group)}
        for col in TOKEN_COLUMNS:
            series = group[col].dropna()
            row[f"{col}_mean"] = round(series.mean(), 2) if not series.empty else pd.NA
            row[f"{col}_median"] = (
                round(series.median(), 2) if not series.empty else pd.NA
            )
            row[f"{col}_min"] = round(series.min(), 2) if not series.empty else pd.NA
            row[f"{col}_max"] = round(series.max(), 2) if not series.empty else pd.NA
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("model")
    return summary_df


def main() -> None:
    loader = DataLoader(base_path=str(ROOT / "datasets"))
    df = loader.load_generation_token_runs()
    if df.empty:
        print("No generation token run data found. Check phase3/datasets.")
        return

    df = df.copy()
    df = df[df["variation"] == "P-3"].copy()
    if df.empty:
        print("No P-3 rows found in generation token runs.")
        return

    df["model_normalized"] = df["model"].map(normalize_model_name)
    df = coerce_numeric(df, TOKEN_COLUMNS)

    summary_df = summarize_tokens(df)

    output_dir = ROOT / "analysis" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rq3_p3_token_summary.csv"
    summary_df.to_csv(output_path, index=False)

    print(f"P-3 rows: {len(df)}")
    print("Token columns summarized:", ", ".join(TOKEN_COLUMNS))
    print(summary_df.to_string(index=False))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
