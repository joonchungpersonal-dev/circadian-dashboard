"""Parse Apple Health XML export (the 'gold standard' full export)."""

import xml.etree.ElementTree as ET
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import pandas as pd
import pytz

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import LOCAL_TIMEZONE

logger = logging.getLogger(__name__)


class HealthKitXMLParser:
    """Parse the full Apple Health XML export."""

    # Metrics we care about
    METRICS_OF_INTEREST = {
        "HKQuantityTypeIdentifierHeartRate": "heart_rate",
        "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "heart_rate_variability",
        "HKQuantityTypeIdentifierRestingHeartRate": "resting_heart_rate",
        "HKQuantityTypeIdentifierRespiratoryRate": "respiratory_rate",
        "HKQuantityTypeIdentifierStepCount": "step_count",
        "HKQuantityTypeIdentifierActiveEnergyBurned": "active_energy",
        "HKQuantityTypeIdentifierBasalEnergyBurned": "basal_energy_burned",
        "HKQuantityTypeIdentifierDistanceWalkingRunning": "walking_running_distance",
        "HKQuantityTypeIdentifierAppleExerciseTime": "apple_exercise_time",
        "HKQuantityTypeIdentifierAppleStandTime": "apple_stand_time",
        "HKQuantityTypeIdentifierVO2Max": "vo2_max",
        "HKCategoryTypeIdentifierSleepAnalysis": "sleep_analysis",
        "HKCategoryTypeIdentifierAppleStandHour": "apple_stand_hour",
    }

    # Sleep analysis value mapping
    SLEEP_VALUES = {
        "HKCategoryValueSleepAnalysisInBed": 0,
        "HKCategoryValueSleepAnalysisAsleepUnspecified": 1,
        "HKCategoryValueSleepAnalysisAsleep": 1,
        "HKCategoryValueSleepAnalysisAsleepCore": 1,
        "HKCategoryValueSleepAnalysisAsleepDeep": 2,
        "HKCategoryValueSleepAnalysisAsleepREM": 3,
        "HKCategoryValueSleepAnalysisAwake": 4,
    }

    def __init__(self):
        self.local_tz = pytz.timezone(LOCAL_TIMEZONE)

    def parse_export(
        self,
        xml_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Parse Apple Health XML export.

        Args:
            xml_path: Path to export.xml file
            start_date: Optional filter start date
            end_date: Optional filter end date
            metrics: Optional list of metric types to include

        Returns:
            DataFrame with columns: timestamp, metric_type, value, unit, source_name, source_bundle_id
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"Export file not found: {xml_path}")

        logger.info(f"Parsing Apple Health XML export: {xml_path}")
        logger.info(f"File size: {xml_path.stat().st_size / 1024 / 1024:.1f} MB")

        records = []
        record_count = 0
        skipped_count = 0

        # Use iterparse for memory efficiency with large files
        context = ET.iterparse(str(xml_path), events=("end",))

        for event, elem in context:
            if elem.tag == "Record":
                record_count += 1

                # Get record type
                record_type = elem.get("type", "")

                # Skip if not a metric we care about
                if record_type not in self.METRICS_OF_INTEREST:
                    skipped_count += 1
                    elem.clear()
                    continue

                metric_name = self.METRICS_OF_INTEREST[record_type]

                # Filter by requested metrics
                if metrics and metric_name not in metrics:
                    skipped_count += 1
                    elem.clear()
                    continue

                # Parse timestamp
                start_date_str = elem.get("startDate", "")
                try:
                    timestamp = pd.to_datetime(start_date_str)
                    if timestamp.tzinfo is None:
                        timestamp = self.local_tz.localize(timestamp)
                    timestamp = timestamp.astimezone(pytz.UTC)
                except Exception:
                    skipped_count += 1
                    elem.clear()
                    continue

                # Date filtering
                if start_date and timestamp < start_date:
                    skipped_count += 1
                    elem.clear()
                    continue
                if end_date and timestamp > end_date:
                    skipped_count += 1
                    elem.clear()
                    continue

                # Get value
                if record_type == "HKCategoryTypeIdentifierSleepAnalysis":
                    # Sleep analysis uses categorical values
                    value_str = elem.get("value", "")
                    value = self.SLEEP_VALUES.get(value_str, 1)
                else:
                    value = float(elem.get("value", 0))

                # Get source
                source_name = elem.get("sourceName", "Unknown")
                source_bundle = elem.get("sourceIdentifier", "")

                # Get unit
                unit = elem.get("unit", "")

                records.append({
                    "timestamp": timestamp,
                    "metric_type": metric_name,
                    "value": value,
                    "unit": unit,
                    "source_name": source_name,
                    "source_bundle_id": source_bundle,
                })

                # Progress logging
                if record_count % 500000 == 0:
                    logger.info(f"Processed {record_count:,} records, kept {len(records):,}")

                # Clear element to free memory
                elem.clear()

        logger.info(f"Parsing complete: {record_count:,} total records, {len(records):,} kept, {skipped_count:,} skipped")

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def parse_and_save_baseline(
        self,
        xml_path: Path,
        output_path: Path,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Parse XML export and save as parquet baseline.

        Args:
            xml_path: Path to export.xml
            output_path: Path for output parquet file
            start_date: Optional start date filter (e.g., last 2 years)

        Returns:
            Parsed DataFrame
        """
        df = self.parse_export(xml_path, start_date=start_date)

        if df.empty:
            logger.warning("No records parsed from XML")
            return df

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as parquet
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved baseline to {output_path} ({len(df):,} records)")

        # Print summary
        print(f"\n=== Baseline Export Summary ===")
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"\nMetric breakdown:")
        print(df["metric_type"].value_counts().to_string())
        print(f"\nSource breakdown:")
        print(df["source_name"].value_counts().head(10).to_string())

        return df


def create_baseline_from_xml(xml_path: str, output_dir: str = None) -> Path:
    """
    Convenience function to create baseline from Apple Health export.

    Args:
        xml_path: Path to export.xml (or the unzipped export folder)
        output_dir: Output directory (defaults to project cache/)

    Returns:
        Path to saved parquet file
    """
    xml_path = Path(xml_path)

    # Handle both direct XML path and export folder
    if xml_path.is_dir():
        xml_path = xml_path / "apple_health_export" / "export.xml"
        if not xml_path.exists():
            xml_path = xml_path.parent.parent / "export.xml"

    if not xml_path.exists():
        raise FileNotFoundError(f"Could not find export.xml at {xml_path}")

    # Default output path
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "cache"

    output_path = Path(output_dir) / "healthkit_baseline.parquet"

    parser = HealthKitXMLParser()

    # Parse last 2 years by default
    from datetime import timedelta
    start_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=730)

    parser.parse_and_save_baseline(xml_path, output_path, start_date=start_date)

    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python healthkit_xml_parser.py <path_to_export.xml_or_export_folder>")
        print("\nExample:")
        print("  python healthkit_xml_parser.py ~/Downloads/export.zip")
        print("  python healthkit_xml_parser.py ~/Downloads/apple_health_export/export.xml")
        sys.exit(1)

    xml_path = sys.argv[1]
    output_path = create_baseline_from_xml(xml_path)
    print(f"\nBaseline saved to: {output_path}")
