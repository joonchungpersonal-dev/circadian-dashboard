"""Load and parse HealthKit data from GCS exports with deduplication.

Supports two data sources:
1. Baseline parquet (from Apple Health XML export) - historical "gold standard"
2. Auto Export JSON (from GCS) - incremental updates

The loader merges both sources and deduplicates.
"""

import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import pytz
from google.cloud import storage

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    GCS_BUCKET_NAME,
    GCS_EXPORTS_PREFIX,
    LOCAL_TIMEZONE,
)

logger = logging.getLogger(__name__)

# Default path for baseline parquet
BASELINE_PATH = Path(__file__).parent.parent.parent / "cache" / "healthkit_baseline.parquet"


class HealthKitLoader:
    """Load HealthKit exports from GCS and parse into DataFrames with deduplication."""

    def __init__(
        self,
        bucket_name: str = GCS_BUCKET_NAME,
        credentials_path: Optional[str] = None
    ):
        """
        Initialize with GCS bucket name and optional credentials.

        Args:
            bucket_name: GCS bucket name containing exports
            credentials_path: Path to service account JSON (optional)
        """
        self.bucket_name = bucket_name
        self.local_tz = pytz.timezone(LOCAL_TIMEZONE)

        if credentials_path:
            self.client = storage.Client.from_service_account_json(credentials_path)
        else:
            self.client = storage.Client()

        self.bucket = self.client.bucket(bucket_name)

    def list_exports(self) -> List[str]:
        """
        List all export blob paths sorted by name (chronological).

        Returns:
            List of all JSON blob paths
        """
        prefix = GCS_EXPORTS_PREFIX
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)

        blob_paths = []
        for blob in blobs:
            if blob.name.endswith(".json"):
                blob_paths.append(blob.name)

        logger.info(f"Found {len(blob_paths)} export files in GCS")
        return sorted(blob_paths)

    def load_export(self, blob_name: str) -> Dict[str, Any]:
        """
        Download and parse a single export file.

        Args:
            blob_name: GCS blob path

        Returns:
            Parsed JSON as dictionary
        """
        blob = self.bucket.blob(blob_name)
        content = blob.download_as_text()
        return json.loads(content)

    def _normalize_source_name(self, source_name: str) -> str:
        """Normalize source name for consistent deduplication."""
        if not source_name:
            return "Unknown"
        # Replace non-breaking spaces, smart quotes, etc.
        normalized = source_name.replace('\xa0', ' ')
        normalized = normalized.replace(''', "'").replace(''', "'")
        normalized = normalized.strip()
        return normalized

    def parse_auto_export_json(self, raw_data: Dict, export_name: str = "") -> pd.DataFrame:
        """
        Parse Auto Export JSON structure into flat DataFrame.

        Args:
            raw_data: Raw JSON data from export
            export_name: Name of export file for logging

        Returns:
            DataFrame with columns:
            - timestamp: datetime (UTC)
            - metric_type: str
            - value: float
            - unit: str
            - source_name: str
            - source_bundle_id: str
        """
        records = []

        # Handle nested structure - data might be under 'data.data' key
        data = raw_data
        if "data" in data:
            data = data["data"]
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        # Get metrics array
        metrics = None
        if isinstance(data, dict) and "metrics" in data:
            metrics = data["metrics"]
        elif isinstance(data, list):
            metrics = data

        if not metrics:
            logger.warning(f"No metrics found in export {export_name}")
            return pd.DataFrame()

        for metric in metrics:
            metric_name = metric.get("name", "unknown")
            unit = metric.get("units", metric.get("unit", ""))

            data_points = metric.get("data", [])
            for point in data_points:
                # Parse timestamp - try multiple field names
                date_str = (point.get("date") or
                           point.get("startDate") or
                           point.get("sleepStart") or "")
                if not date_str:
                    continue

                try:
                    timestamp = pd.to_datetime(date_str)
                    if timestamp.tzinfo is None:
                        timestamp = self.local_tz.localize(timestamp)
                    timestamp = timestamp.astimezone(pytz.UTC)
                except Exception as e:
                    logger.debug(f"Could not parse date '{date_str}': {e}")
                    continue

                # Get and normalize source info
                source_name = point.get("source", point.get("sourceName", "Unknown"))
                source_name = self._normalize_source_name(source_name)
                source_bundle = point.get("sourceBundle", point.get("sourceBundleId", ""))

                # Special handling for sleep_analysis - calculate duration from timestamps
                # instead of trusting totalSleep (which can be corrupted/cumulative)
                if metric_name == "sleep_analysis" and "sleepStart" in point and "sleepEnd" in point:
                    try:
                        sleep_start = pd.to_datetime(point["sleepStart"])
                        if sleep_start.tzinfo is None:
                            sleep_start = self.local_tz.localize(sleep_start)
                        sleep_start = sleep_start.astimezone(pytz.UTC)

                        sleep_end = pd.to_datetime(point["sleepEnd"])
                        if sleep_end.tzinfo is None:
                            sleep_end = self.local_tz.localize(sleep_end)
                        sleep_end = sleep_end.astimezone(pytz.UTC)

                        # Calculate actual duration from timestamps
                        duration_hours = (sleep_end - sleep_start).total_seconds() / 3600

                        # Skip invalid sessions (too short or impossibly long)
                        if duration_hours < 0.5 or duration_hours > 14:
                            logger.debug(f"Skipping invalid sleep session: {duration_hours:.1f} hours")
                            continue

                        # Add summary record with CALCULATED duration (not totalSleep)
                        records.append({
                            "timestamp": timestamp,
                            "metric_type": metric_name,
                            "value": float(duration_hours),
                            "unit": "hours",
                            "source_name": source_name,
                            "source_bundle_id": source_bundle,
                        })

                        # Add sleep start/end markers
                        records.append({
                            "timestamp": sleep_start,
                            "metric_type": "sleep_analysis",
                            "value": 1.0,
                            "unit": "",
                            "source_name": source_name,
                            "source_bundle_id": source_bundle,
                        })
                        records.append({
                            "timestamp": sleep_end,
                            "metric_type": "sleep_analysis",
                            "value": 1.0,
                            "unit": "",
                            "source_name": source_name,
                            "source_bundle_id": source_bundle,
                        })
                    except Exception as e:
                        logger.debug(f"Could not parse sleep times: {e}")
                else:
                    # For non-sleep metrics, use the standard value extraction
                    value = (point.get("qty") or
                            point.get("value") or
                            point.get("Avg") or
                            point.get("totalSleep") or
                            point.get("asleep") or
                            point.get("inBed") or 0)

                    records.append({
                        "timestamp": timestamp,
                        "metric_type": metric_name,
                        "value": float(value) if value is not None else 0.0,
                        "unit": unit,
                        "source_name": source_name,
                        "source_bundle_id": source_bundle,
                    })

        df = pd.DataFrame(records)

        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            logger.info(f"Parsed {len(df)} records from {export_name}")

        return df

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate records from overlapping exports.

        Strategy:
        1. Round timestamps to seconds to handle precision differences
        2. Create composite key from (timestamp_rounded, metric_type, source_name, value_rounded)
        3. Keep the last occurrence (most recent export wins)

        Args:
            df: Combined DataFrame from multiple exports

        Returns:
            Deduplicated DataFrame
        """
        if df.empty:
            return df

        before_count = len(df)

        # Round timestamp to seconds for deduplication
        df["_ts_rounded"] = df["timestamp"].dt.floor("s")

        # Round value to 2 decimal places for deduplication
        df["_value_rounded"] = df["value"].round(2)

        # Deduplicate on rounded keys
        df = df.drop_duplicates(
            subset=["_ts_rounded", "metric_type", "source_name", "_value_rounded"],
            keep="last"
        )

        # Clean up temp columns
        df = df.drop(columns=["_ts_rounded", "_value_rounded"])

        after_count = len(df)
        duplicates_removed = before_count - after_count

        if duplicates_removed > 0:
            logger.info(f"Deduplication: {before_count} → {after_count} records ({duplicates_removed} duplicates removed)")

        return df

    def load_baseline(self, baseline_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load baseline parquet file (from Apple Health XML export).

        Args:
            baseline_path: Path to baseline parquet (defaults to cache/healthkit_baseline.parquet)

        Returns:
            DataFrame or empty DataFrame if baseline doesn't exist
        """
        path = baseline_path or BASELINE_PATH

        if not path.exists():
            logger.info(f"No baseline file found at {path}")
            return pd.DataFrame()

        try:
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            logger.info(f"Loaded baseline: {len(df):,} records from {path}")
            return df
        except Exception as e:
            logger.error(f"Error loading baseline: {e}")
            return pd.DataFrame()

    def load_date_range(
        self,
        start_date: date,
        end_date: date,
        metrics: Optional[List[str]] = None,
        max_exports: Optional[int] = None,
        use_baseline: bool = True
    ) -> pd.DataFrame:
        """
        Load and merge data from baseline + Auto Export with deduplication.

        Data flow:
        1. Load baseline parquet (historical "gold standard" from XML export)
        2. Load recent Auto Export JSON files (incremental updates)
        3. Merge and deduplicate (baseline wins for historical, Auto Export for recent)
        4. Filter to requested date range

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            metrics: Optional list of metric types to filter
            max_exports: Max Auto Export files to load (default 10)
            use_baseline: Whether to include baseline data (default True)

        Returns:
            Combined, deduplicated DataFrame
        """
        all_dfs = []

        # 1. Load baseline (historical data from XML export)
        if use_baseline:
            baseline_df = self.load_baseline()
            if not baseline_df.empty:
                all_dfs.append(baseline_df)

        # 2. Load Auto Export data (recent updates)
        blob_paths = self.list_exports()

        if blob_paths:
            # Default to 10 most recent exports
            if max_exports is None:
                max_exports = 10

            if len(blob_paths) > max_exports:
                paths_to_load = blob_paths[-max_exports:]
                logger.info(f"Loading {len(paths_to_load)} most recent Auto Exports (of {len(blob_paths)} total)")
            else:
                paths_to_load = blob_paths
                logger.info(f"Loading all {len(paths_to_load)} Auto Exports")

            for blob_path in paths_to_load:
                try:
                    raw_data = self.load_export(blob_path)
                    df = self.parse_auto_export_json(raw_data, export_name=blob_path.split("/")[-1])
                    if not df.empty:
                        all_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error loading {blob_path}: {e}")

        if not all_dfs:
            logger.warning("No data found from baseline or Auto Export")
            return pd.DataFrame()

        # 3. Combine all sources
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Combined {len(combined):,} records from {len(all_dfs)} sources (before deduplication)")

        # 4. Deduplicate (most recent source wins for duplicates)
        combined = self._deduplicate(combined)

        # 5. Filter by date range
        if start_date:
            start_dt = pd.Timestamp(start_date, tz='UTC')
            combined = combined[combined["timestamp"] >= start_dt]
        if end_date:
            end_dt = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
            combined = combined[combined["timestamp"] < end_dt]

        # 6. Filter by metrics if specified
        if metrics:
            combined = combined[combined["metric_type"].isin(metrics)]

        # Sort by timestamp
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Final: {len(combined):,} records for {start_date} to {end_date}")
        return combined

    def load_all_data(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load ALL data from ALL exports with full deduplication.

        Use this for initial data loading or when you need complete historical data.

        Args:
            metrics: Optional list of metric types to filter

        Returns:
            Complete, deduplicated DataFrame
        """
        return self.load_date_range(
            start_date=date(2020, 1, 1),
            end_date=date.today(),
            metrics=metrics,
            max_exports=None  # Load all
        )


def load_healthkit_data(
    start_date: date,
    end_date: date,
    bucket_name: str = GCS_BUCKET_NAME
) -> pd.DataFrame:
    """
    Convenience function to load HealthKit data.

    Args:
        start_date: Start date
        end_date: End date
        bucket_name: GCS bucket name

    Returns:
        Combined DataFrame of all HealthKit data
    """
    loader = HealthKitLoader(bucket_name=bucket_name)
    return loader.load_date_range(start_date, end_date)
