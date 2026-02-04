"""Compare sleep metrics between Apple Watch and Eight Sleep."""

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    APPLE_WATCH_SOURCE_PATTERNS,
    EIGHT_SLEEP_SOURCE_PATTERNS,
    NIGHT_BOUNDARY_HOUR,
    MIN_SLEEP_DURATION_MINUTES,
    MAX_SLEEP_GAP_MINUTES,
    LOCAL_TIMEZONE,
    METRIC_SLEEP_ANALYSIS,
    METRIC_HEART_RATE,
    METRIC_HRV,
)

logger = logging.getLogger(__name__)


@dataclass
class NightDefinition:
    """Define what constitutes a 'night' for sleep attribution."""

    boundary_hour: int = NIGHT_BOUNDARY_HOUR  # 6 PM
    min_sleep_duration_minutes: int = MIN_SLEEP_DURATION_MINUTES
    max_gap_minutes: int = MAX_SLEEP_GAP_MINUTES


class DeviceComparator:
    """Compare sleep metrics between Apple Watch and Eight Sleep."""

    def __init__(
        self,
        df: pd.DataFrame,
        night_definition: Optional[NightDefinition] = None
    ):
        """
        Initialize with raw HealthKit DataFrame.

        Args:
            df: DataFrame with columns [timestamp, metric_type, value, unit, source_name]
            night_definition: Custom night boundary definition
        """
        self.df = df.copy()
        self.night_def = night_definition or NightDefinition()
        self.local_tz = pytz.timezone(LOCAL_TIMEZONE)

        # Ensure timestamp is datetime
        if not self.df.empty:
            self.df["timestamp"] = pd.to_datetime(self.df["timestamp"], utc=True)

        # Classify sources
        self.df["device"] = self.df["source_name"].apply(self._classify_source)

    def _classify_source(self, source_name: str) -> str:
        """
        Classify a source_name as 'apple_watch', 'eight_sleep', or 'other'.

        Args:
            source_name: Source name string

        Returns:
            Device classification
        """
        if not source_name:
            return "other"

        source_lower = source_name.lower()

        for pattern in APPLE_WATCH_SOURCE_PATTERNS:
            if pattern.lower() in source_lower:
                return "apple_watch"

        for pattern in EIGHT_SLEEP_SOURCE_PATTERNS:
            if pattern.lower() in source_lower:
                return "eight_sleep"

        return "other"

    def identify_sources(self) -> Dict[str, List[str]]:
        """
        Auto-detect device sources in the data.

        Returns:
            Dictionary mapping device type to list of source names
        """
        sources = {"apple_watch": [], "eight_sleep": [], "other": []}

        for source in self.df["source_name"].unique():
            device = self._classify_source(source)
            if source not in sources[device]:
                sources[device].append(source)

        return sources

    def _assign_night_date(self, timestamp: datetime) -> date:
        """
        Assign a timestamp to a 'night date'.

        Logic: If time is before boundary_hour (6 PM), assign to previous date.
        Example: 2024-01-15 02:30 AM → night of 2024-01-14
                 2024-01-15 10:30 PM → night of 2024-01-15

        Args:
            timestamp: UTC timestamp

        Returns:
            Night date
        """
        # Convert to local time
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        local_time = timestamp.astimezone(self.local_tz)

        # If before boundary hour, assign to previous day
        if local_time.hour < self.night_def.boundary_hour:
            return (local_time - timedelta(days=1)).date()
        return local_time.date()

    def extract_sleep_sessions(
        self,
        device: str = "all"
    ) -> pd.DataFrame:
        """
        Extract and consolidate sleep sessions.

        Handles two data formats:
        1. Marker records (value=1.0): Use timestamps as sleep start/end
        2. Summary records (value>1): Value contains duration in hours

        Args:
            device: Filter by device ('apple_watch', 'eight_sleep', or 'all')

        Returns:
            DataFrame with sleep session info per night
        """
        # Filter to sleep_analysis data only (not temperature)
        sleep_df = self.df[
            self.df["metric_type"] == "sleep_analysis"
        ].copy()

        if sleep_df.empty:
            logger.warning("No sleep_analysis data found")
            return pd.DataFrame()

        # Filter by device if specified
        if device != "all":
            sleep_df = sleep_df[sleep_df["device"] == device]

        if sleep_df.empty:
            return pd.DataFrame()

        # Assign night dates
        sleep_df["night_date"] = sleep_df["timestamp"].apply(self._assign_night_date)

        # Group by night and device
        sessions = []
        for (night, dev), group in sleep_df.groupby(["night_date", "device"]):
            if len(group) == 0:
                continue

            group = group.sort_values("timestamp")

            # Separate markers (value=1.0) from duration summaries (value>1)
            markers = group[group["value"] == 1.0]
            summaries = group[group["value"] > 1.0]

            sleep_start = None
            sleep_end = None
            duration = None

            # If we have markers, use them for start/end times
            if len(markers) >= 2:
                sleep_start = markers["timestamp"].min()
                sleep_end = markers["timestamp"].max()
                duration = (sleep_end - sleep_start).total_seconds() / 3600

            # If we have a summary record, use its value as duration
            if not summaries.empty:
                summary_duration = summaries["value"].max()
                if duration is None or summary_duration > 0:
                    duration = summary_duration
                # Estimate start/end from summary timestamp if no markers
                if sleep_start is None:
                    summary_time = summaries["timestamp"].iloc[0]
                    # Summary typically posted at 5 AM, estimate sleep window
                    sleep_end = summary_time + pd.Timedelta(hours=duration / 2)
                    sleep_start = summary_time - pd.Timedelta(hours=duration / 2)

            # Fallback: use all records' timestamp span
            if sleep_start is None:
                sleep_start = group["timestamp"].min()
                sleep_end = group["timestamp"].max()
                if duration is None:
                    duration = (sleep_end - sleep_start).total_seconds() / 3600

            # Skip invalid sessions
            if duration is None or duration < self.night_def.min_sleep_duration_minutes / 60:
                continue

            # Calculate midpoint
            midpoint = sleep_start + (sleep_end - sleep_start) / 2

            sessions.append({
                "night_date": night,
                "device": dev,
                "source_name": group["source_name"].iloc[0],
                "sleep_start": sleep_start,
                "sleep_end": sleep_end,
                "duration_hours": duration,
                "midpoint": midpoint,
                "samples": len(group),
            })

        result = pd.DataFrame(sessions)
        if not result.empty:
            result = result.sort_values("night_date")

        return result

    def extract_hrv(self, device: str = "all") -> pd.DataFrame:
        """
        Extract nightly HRV statistics.

        Args:
            device: Filter by device

        Returns:
            DataFrame with nightly HRV stats
        """
        hrv_df = self.df[
            self.df["metric_type"].str.contains("hrv|variability", case=False, na=False)
        ].copy()

        if hrv_df.empty:
            return pd.DataFrame()

        if device != "all":
            hrv_df = hrv_df[hrv_df["device"] == device]

        hrv_df["night_date"] = hrv_df["timestamp"].apply(self._assign_night_date)

        # Aggregate by night and device
        stats = hrv_df.groupby(["night_date", "device"]).agg(
            hrv_mean=("value", "mean"),
            hrv_median=("value", "median"),
            hrv_min=("value", "min"),
            hrv_max=("value", "max"),
            hrv_samples=("value", "count"),
        ).reset_index()

        return stats

    def extract_hrv_unified(self) -> pd.DataFrame:
        """
        Extract nightly HRV statistics without device filtering.
        Use this when HRV data lacks source attribution.

        Returns:
            DataFrame with nightly HRV stats (all sources combined)
        """
        hrv_df = self.df[
            self.df["metric_type"].str.contains("hrv|variability", case=False, na=False)
        ].copy()

        if hrv_df.empty:
            return pd.DataFrame()

        hrv_df["night_date"] = hrv_df["timestamp"].apply(self._assign_night_date)

        # Aggregate by night only (ignore device)
        stats = hrv_df.groupby("night_date").agg(
            hrv_mean=("value", "mean"),
            hrv_median=("value", "median"),
            hrv_min=("value", "min"),
            hrv_max=("value", "max"),
            hrv_samples=("value", "count"),
        ).reset_index()

        return stats

    def extract_heart_rate(self, device: str = "all") -> pd.DataFrame:
        """
        Extract nightly heart rate statistics.

        Args:
            device: Filter by device

        Returns:
            DataFrame with nightly HR stats
        """
        hr_df = self.df[
            self.df["metric_type"].str.contains("heart_rate|heartrate", case=False, na=False) &
            ~self.df["metric_type"].str.contains("variability", case=False, na=False)
        ].copy()

        if hr_df.empty:
            return pd.DataFrame()

        if device != "all":
            hr_df = hr_df[hr_df["device"] == device]

        hr_df["night_date"] = hr_df["timestamp"].apply(self._assign_night_date)

        # Aggregate by night and device
        def resting_hr(x):
            """Calculate resting HR as 5th percentile."""
            return np.percentile(x, 5)

        stats = hr_df.groupby(["night_date", "device"]).agg(
            hr_mean=("value", "mean"),
            hr_min=("value", "min"),
            hr_max=("value", "max"),
            hr_resting=("value", resting_hr),
            hr_samples=("value", "count"),
        ).reset_index()

        return stats

    def extract_heart_rate_by_device(self) -> pd.DataFrame:
        """
        Extract nightly heart rate statistics split by device.
        Returns data even if only one device has data.

        Returns:
            DataFrame with nightly HR stats per device (wide format)
        """
        hr_df = self.df[
            self.df["metric_type"].str.contains("heart_rate|heartrate", case=False, na=False) &
            ~self.df["metric_type"].str.contains("variability", case=False, na=False)
        ].copy()

        if hr_df.empty:
            return pd.DataFrame()

        hr_df["night_date"] = hr_df["timestamp"].apply(self._assign_night_date)

        def resting_hr(x):
            return np.percentile(x, 5)

        # Aggregate by night and device
        stats = hr_df.groupby(["night_date", "device"]).agg(
            hr_mean=("value", "mean"),
            hr_min=("value", "min"),
            hr_max=("value", "max"),
            hr_resting=("value", resting_hr),
            hr_samples=("value", "count"),
        ).reset_index()

        # Pivot to wide format
        result = stats.pivot(index="night_date", columns="device", values=["hr_mean", "hr_min", "hr_max", "hr_resting"])
        result.columns = [f"{col[0]}_{col[1]}" for col in result.columns]
        result = result.reset_index()

        return result

    def build_comparison_df(self) -> pd.DataFrame:
        """
        Build the master comparison DataFrame.

        One row per night where BOTH devices have data.

        Returns:
            Comparison DataFrame with metrics from both devices
        """
        # Get sleep sessions for each device
        watch_sleep = self.extract_sleep_sessions("apple_watch")
        eight_sleep = self.extract_sleep_sessions("eight_sleep")

        if watch_sleep.empty or eight_sleep.empty:
            logger.warning("Not enough data from both devices for comparison")
            return pd.DataFrame()

        # Merge on night_date
        comparison = pd.merge(
            watch_sleep,
            eight_sleep,
            on="night_date",
            suffixes=("_watch", "_eight"),
            how="inner"  # Only nights with both devices
        )

        if comparison.empty:
            return pd.DataFrame()

        # Calculate timing differences (in minutes)
        comparison["diff_sleep_start_min"] = (
            comparison["sleep_start_watch"] - comparison["sleep_start_eight"]
        ).dt.total_seconds() / 60

        comparison["diff_sleep_end_min"] = (
            comparison["sleep_end_watch"] - comparison["sleep_end_eight"]
        ).dt.total_seconds() / 60

        comparison["diff_duration_min"] = (
            comparison["duration_hours_watch"] - comparison["duration_hours_eight"]
        ) * 60

        comparison["diff_midpoint_min"] = (
            comparison["midpoint_watch"] - comparison["midpoint_eight"]
        ).dt.total_seconds() / 60

        # Add HRV data
        watch_hrv = self.extract_hrv("apple_watch")
        eight_hrv = self.extract_hrv("eight_sleep")

        if not watch_hrv.empty and not eight_hrv.empty:
            hrv_comparison = pd.merge(
                watch_hrv,
                eight_hrv,
                on="night_date",
                suffixes=("_watch", "_eight"),
                how="inner"
            )
            if not hrv_comparison.empty:
                comparison = pd.merge(
                    comparison,
                    hrv_comparison[["night_date", "hrv_mean_watch", "hrv_mean_eight",
                                   "hrv_min_watch", "hrv_min_eight",
                                   "hrv_max_watch", "hrv_max_eight"]],
                    on="night_date",
                    how="left"
                )
                comparison["diff_hrv"] = (
                    comparison["hrv_mean_watch"] - comparison["hrv_mean_eight"]
                )

        # Add HR data
        watch_hr = self.extract_heart_rate("apple_watch")
        eight_hr = self.extract_heart_rate("eight_sleep")

        if not watch_hr.empty and not eight_hr.empty:
            hr_comparison = pd.merge(
                watch_hr,
                eight_hr,
                on="night_date",
                suffixes=("_watch", "_eight"),
                how="inner"
            )
            if not hr_comparison.empty:
                comparison = pd.merge(
                    comparison,
                    hr_comparison[["night_date", "hr_min_watch", "hr_min_eight",
                                  "hr_max_watch", "hr_max_eight",
                                  "hr_resting_watch", "hr_resting_eight"]],
                    on="night_date",
                    how="left"
                )
                comparison["diff_hr_resting"] = (
                    comparison.get("hr_resting_watch", 0) -
                    comparison.get("hr_resting_eight", 0)
                )

        # Calculate agreement score
        comparison["agreement_score"] = comparison.apply(
            self.calculate_agreement_score, axis=1
        )

        # Sort by date
        comparison = comparison.sort_values("night_date").reset_index(drop=True)

        return comparison

    def calculate_agreement_score(self, row: pd.Series) -> float:
        """
        Calculate composite agreement score (0-100).

        Weights:
        - Sleep timing (start, end, midpoint): 50%
        - Duration: 20%
        - HRV: 15%
        - Heart rate: 15%

        Args:
            row: DataFrame row

        Returns:
            Agreement score 0-100
        """
        scores = []
        weights = []

        # Timing scores (penalize by minutes difference)
        def timing_score(diff_min: float, max_diff: float = 60) -> float:
            """Convert minute difference to 0-100 score."""
            if pd.isna(diff_min):
                return 50  # Neutral if missing
            diff = abs(diff_min)
            return max(0, 100 - (diff / max_diff * 100))

        # Sleep start
        if "diff_sleep_start_min" in row:
            scores.append(timing_score(row["diff_sleep_start_min"], 30))
            weights.append(0.15)

        # Sleep end
        if "diff_sleep_end_min" in row:
            scores.append(timing_score(row["diff_sleep_end_min"], 30))
            weights.append(0.15)

        # Midpoint
        if "diff_midpoint_min" in row:
            scores.append(timing_score(row["diff_midpoint_min"], 30))
            weights.append(0.20)

        # Duration
        if "diff_duration_min" in row:
            scores.append(timing_score(row["diff_duration_min"], 60))
            weights.append(0.20)

        # HRV
        if "diff_hrv" in row and not pd.isna(row.get("diff_hrv")):
            hrv_score = max(0, 100 - abs(row["diff_hrv"]) * 2)  # 50ms diff = 0
            scores.append(hrv_score)
            weights.append(0.15)

        # HR
        if "diff_hr_resting" in row and not pd.isna(row.get("diff_hr_resting")):
            hr_score = max(0, 100 - abs(row["diff_hr_resting"]) * 5)  # 20bpm diff = 0
            scores.append(hr_score)
            weights.append(0.15)

        if not scores:
            return 50.0

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 50.0

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return round(weighted_score, 1)

    def get_summary_statistics(self) -> Dict:
        """
        Calculate overall comparison statistics.

        Returns:
            Dictionary of summary statistics
        """
        watch_sleep = self.extract_sleep_sessions("apple_watch")
        eight_sleep = self.extract_sleep_sessions("eight_sleep")
        comparison = self.build_comparison_df()

        watch_nights = set(watch_sleep["night_date"]) if not watch_sleep.empty else set()
        eight_nights = set(eight_sleep["night_date"]) if not eight_sleep.empty else set()

        stats = {
            "total_nights": len(watch_nights | eight_nights),
            "nights_both_devices": len(watch_nights & eight_nights),
            "nights_watch_only": len(watch_nights - eight_nights),
            "nights_eight_sleep_only": len(eight_nights - watch_nights),
        }

        if not comparison.empty:
            stats["mean_duration_diff_min"] = comparison["diff_duration_min"].mean()
            stats["mean_midpoint_diff_min"] = comparison["diff_midpoint_min"].mean()
            stats["mean_agreement_score"] = comparison["agreement_score"].mean()

            # Correlations
            if "duration_hours_watch" in comparison and "duration_hours_eight" in comparison:
                valid = comparison[["duration_hours_watch", "duration_hours_eight"]].dropna()
                if len(valid) > 2:
                    stats["duration_correlation"] = valid["duration_hours_watch"].corr(
                        valid["duration_hours_eight"]
                    )

            if "hrv_mean_watch" in comparison and "hrv_mean_eight" in comparison:
                valid = comparison[["hrv_mean_watch", "hrv_mean_eight"]].dropna()
                if len(valid) > 2:
                    stats["hrv_correlation"] = valid["hrv_mean_watch"].corr(
                        valid["hrv_mean_eight"]
                    )

        return stats
