"""
Detection Utilities Module

This module provides shared utilities for the multi-detector system:
- DetectionLogger: Enhanced logging with levels, detector tags, and performance metrics
- ProgressTracker: Detector-specific progress tracking with metadata
- CSVHandler: Dynamic CSV column management for multiple detectors

These utilities ensure consistent logging, resumability, and data management
across all detection tools in the research pipeline.
"""

import datetime
import os
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


class DetectionLogger:
    """
    Enhanced logging system with levels and detector tags.

    Features:
    - Multiple log levels (INFO, SUCCESS, WARNING, ERROR, PERF)
    - Detector-specific log files
    - Master log file for all detectors
    - Performance metrics tracking
    - Console output synchronization
    """

    LOG_LEVELS = ["INFO", "SUCCESS", "WARNING", "ERROR", "PERF"]

    def __init__(
        self, log_dir: str, master_log_name: str = "detection_master_output.log"
    ):
        """
        Initialize the logger.

        Args:
            log_dir: Directory where log files will be stored
            master_log_name: Name of the master log file
        """
        self.log_dir = log_dir
        self.master_log_path = os.path.join(log_dir, master_log_name)

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

        # Track performance metrics per detector
        self.performance_metrics: Dict[str, List[float]] = {}

    def _get_detector_log_path(self, detector: str) -> str:
        """Get the log file path for a specific detector."""
        return os.path.join(self.log_dir, f"detection_{detector}_output.log")

    def _format_message(self, level: str, detector: str, message: str) -> str:
        """Format a log message with timestamp, level, and detector tag."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} [{level:7}] [{detector:8}] {message}"

    def log(
        self,
        level: str,
        detector: str,
        message: str,
        log_to_master: bool = True,
        console: bool = True,
    ):
        """
        Log a message with specified level and detector tag.

        Args:
            level: Log level (INFO, SUCCESS, WARNING, ERROR, PERF)
            detector: Detector name (e.g., 'zerogpt', 'roberta')
            message: Message to log
            log_to_master: Whether to also log to master log file
            console: Whether to print to console
        """
        if level not in self.LOG_LEVELS:
            level = "INFO"

        formatted_message = self._format_message(level, detector, message)

        # Console output
        if console:
            print(formatted_message)

        # Write to detector-specific log
        detector_log_path = self._get_detector_log_path(detector)
        with open(detector_log_path, "a") as f:
            f.write(formatted_message + "\n")

        # Write to master log
        if log_to_master:
            with open(self.master_log_path, "a") as f:
                f.write(formatted_message + "\n")

    def log_performance(self, detector: str, metric_name: str, value: float):
        """
        Log a performance metric.

        Args:
            detector: Detector name
            metric_name: Name of the metric (e.g., 'entry_time', 'batch_time')
            value: Metric value in seconds
        """
        key = f"{detector}_{metric_name}"
        if key not in self.performance_metrics:
            self.performance_metrics[key] = []

        self.performance_metrics[key].append(value)

        # Log the metric
        self.log("PERF", detector, f"{metric_name}: {value:.2f}s")

    def log_batch_summary(
        self,
        detector: str,
        batch_num: int,
        batch_size: int,
        successful: int,
        failed: int,
        batch_time: float,
    ):
        """
        Log a summary of batch processing.

        Args:
            detector: Detector name
            batch_num: Batch number
            batch_size: Size of the batch
            successful: Number of successful entries
            failed: Number of failed entries
            batch_time: Time taken for the batch in seconds
        """
        avg_time = batch_time / batch_size if batch_size > 0 else 0

        summary = (
            f"Batch {batch_num} complete: "
            f"{successful} successful, {failed} failed, "
            f"{batch_time:.1f}s total ({avg_time:.2f}s/entry)"
        )

        self.log("INFO", detector, summary)

    def get_performance_summary(self, detector: str) -> Dict[str, float]:
        """
        Get performance statistics for a detector.

        Args:
            detector: Detector name

        Returns:
            Dictionary with performance statistics
        """
        metrics = {}

        for key, values in self.performance_metrics.items():
            if key.startswith(f"{detector}_"):
                metric_name = key.replace(f"{detector}_", "")
                if values:
                    metrics[f"{metric_name}_avg"] = sum(values) / len(values)
                    metrics[f"{metric_name}_total"] = sum(values)
                    metrics[f"{metric_name}_count"] = len(values)

        return metrics


class ProgressTracker:
    """
    Detector-specific progress tracking with metadata.

    Features:
    - Independent progress files per detector
    - Metadata tracking (model version, timestamps, statistics)
    - Resumability support
    - Failed entry tracking
    """

    def __init__(self, progress_dir: str):
        """
        Initialize the progress tracker.

        Args:
            progress_dir: Directory where progress files will be stored
        """
        self.progress_dir = progress_dir
        os.makedirs(progress_dir, exist_ok=True)

    def _get_progress_path(self, detector: str) -> str:
        """Get the progress file path for a specific detector."""
        return os.path.join(self.progress_dir, f"detection_progress_{detector}.pkl")

    def save_progress(
        self,
        detector: str,
        df: pd.DataFrame,
        processed_entries: Set[str],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save progress for a detector.

        Args:
            detector: Detector name
            df: DataFrame with current results
            processed_entries: Set of processed entry keys
            metadata: Additional metadata to save
        """
        progress_path = self._get_progress_path(detector)

        # Build metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "last_run": datetime.datetime.now().isoformat(),
                "total_processed": len(processed_entries),
            }
        )

        # Save to pickle
        progress_data = {
            "df": df.to_dict(orient="records"),
            "processed_entries": list(processed_entries),
            "metadata": metadata,
        }

        with open(progress_path, "wb") as f:
            pickle.dump(progress_data, f)

    def load_progress(
        self, detector: str
    ) -> Tuple[pd.DataFrame, Set[str], Dict[str, Any]]:
        """
        Load progress for a detector.

        Args:
            detector: Detector name

        Returns:
            Tuple of (DataFrame, processed_entries set, metadata dict)
        """
        progress_path = self._get_progress_path(detector)

        if not os.path.exists(progress_path):
            return pd.DataFrame(), set(), {}

        try:
            with open(progress_path, "rb") as f:
                progress_data = pickle.load(f)

            # Validate structure
            if not isinstance(progress_data, dict):
                return pd.DataFrame(), set(), {}

            df = pd.DataFrame(progress_data.get("df", []))
            processed_entries = set(progress_data.get("processed_entries", []))
            metadata = progress_data.get("metadata", {})

            return df, processed_entries, metadata

        except Exception as e:
            print(f"Warning: Could not load progress for {detector}: {e}")
            return pd.DataFrame(), set(), {}

    def get_unprocessed_entries(
        self, detector: str, all_entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get entries that haven't been processed by this detector.

        Args:
            detector: Detector name
            all_entries: List of all entry dictionaries with 'entry_key' field

        Returns:
            List of unprocessed entry dictionaries
        """
        df, _, _ = self.load_progress(detector)

        successful_keys = set()
        if not df.empty and "success" in df.columns:
            # Filter for successful entries
            # Ensure boolean comparison handles potential None/NaNs safely
            successful_keys = set(df[df["success"] == True]["entry_key"])

        unprocessed = [
            entry
            for entry in all_entries
            if entry["entry_key"] not in successful_keys
        ]

        return unprocessed

    def clear_progress(self, detector: str):
        """
        Clear progress file for a detector (for force reprocessing).

        Args:
            detector: Detector name
        """
        progress_path = self._get_progress_path(detector)
        if os.path.exists(progress_path):
            os.remove(progress_path)


class CSVHandler:
    """
    Dynamic CSV column management for multiple detectors.

    Features:
    - Dynamic column addition per detector
    - Result merging across detectors
    - Schema validation
    - Deduplication
    """

    def __init__(self, output_csv_path: str):
        """
        Initialize the CSV handler.

        Args:
            output_csv_path: Path to the output CSV file
        """
        self.output_csv_path = output_csv_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    def get_detector_columns(self, detector: str) -> List[str]:
        """
        Get the column names for a specific detector.

        Args:
            detector: Detector name

        Returns:
            List of column names
        """
        return [
            f"{detector}_ai_probability",
            f"{detector}_is_ai",
            f"{detector}_response",
            f"{detector}_success",
        ]

    def load_or_create_df(self) -> pd.DataFrame:
        """
        Load existing CSV or create new DataFrame.

        Returns:
            DataFrame with existing data or empty with base columns
        """
        if os.path.exists(self.output_csv_path):
            try:
                df = pd.read_csv(self.output_csv_path, encoding="utf-8")
                return df
            except Exception as e:
                print(f"Warning: Could not load existing CSV: {e}")

        # Create new DataFrame with base columns
        return pd.DataFrame(
            columns=["pr_id", "prompt_variation", "entry_type", "entry_key"]
        )

    def add_detector_columns(self, df: pd.DataFrame, detector: str) -> pd.DataFrame:
        """
        Add columns for a detector if they don't exist.

        Args:
            df: DataFrame to modify
            detector: Detector name

        Returns:
            Modified DataFrame
        """
        detector_columns = self.get_detector_columns(detector)

        for col in detector_columns:
            if col not in df.columns:
                df[col] = None

        return df

    def merge_detector_results(
        self, df: pd.DataFrame, detector: str, results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Merge detector results into the DataFrame.

        Args:
            df: Existing DataFrame
            detector: Detector name
            results: List of result dictionaries with detector outputs

        Returns:
            Updated DataFrame
        """
        # Ensure detector columns exist
        df = self.add_detector_columns(df, detector)

        # Create a results lookup by entry_key
        results_lookup = {r["entry_key"]: r for r in results}

        # Update rows with detector results
        for idx, row in df.iterrows():
            entry_key = row["entry_key"]
            if entry_key in results_lookup:
                result = results_lookup[entry_key]
                df.loc[idx, f"{detector}_ai_probability"] = result.get("ai_probability")  # type: ignore
                df.loc[idx, f"{detector}_is_ai"] = result.get("is_ai_generated")  # type: ignore
                df.loc[idx, f"{detector}_response"] = result.get("raw_response")  # type: ignore
                df.loc[idx, f"{detector}_success"] = result.get("success", True)  # type: ignore

        return df

    def save_df(self, df: pd.DataFrame, deduplicate: bool = True):
        """
        Save DataFrame to CSV.

        Args:
            df: DataFrame to save
            deduplicate: Whether to remove duplicates based on entry_key
        """
        if deduplicate:
            df = df.drop_duplicates(subset="entry_key", keep="last")

        df.to_csv(self.output_csv_path, index=False, encoding="utf-8")

    def validate_output_schema(self) -> Dict[str, Any]:
        """
        Validate the output CSV schema and return statistics.

        Returns:
            Dictionary with validation statistics
        """
        if not os.path.exists(self.output_csv_path):
            return {"error": "Output CSV does not exist"}

        df = pd.read_csv(self.output_csv_path)

        # Identify detectors from columns
        detectors = set()
        for col in df.columns:
            if col.endswith("_ai_probability"):
                detector = col.replace("_ai_probability", "")
                detectors.add(detector)

        stats = {
            "total_rows": len(df),
            "unique_prs": df["pr_id"].nunique() if "pr_id" in df.columns else 0,
            "detectors": list(detectors),
            "columns": list(df.columns),
        }

        # Per-detector statistics
        for detector in detectors:
            success_col = f"{detector}_success"
            if success_col in df.columns:
                stats[f"{detector}_successful"] = df[success_col].sum()
                stats[f"{detector}_failed"] = len(df) - df[success_col].sum()

        return stats
