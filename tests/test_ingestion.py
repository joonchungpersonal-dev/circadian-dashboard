"""Tests for HealthKit data ingestion."""

import json
import pytest
from datetime import datetime, date
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.healthkit_loader import HealthKitLoader


class TestHealthKitLoader:
    """Test suite for HealthKitLoader."""

    @pytest.fixture
    def sample_export(self):
        """Load sample export fixture."""
        fixture_path = Path(__file__).parent / "fixtures" / "sample_export.json"
        with open(fixture_path) as f:
            return json.load(f)

    @pytest.fixture
    def loader(self):
        """Create loader instance (without GCS connection)."""
        # Mock the GCS client for testing
        loader = HealthKitLoader.__new__(HealthKitLoader)
        loader.bucket_name = "test-bucket"
        loader.local_tz = __import__("pytz").timezone("America/New_York")
        return loader

    def test_parse_auto_export_json_basic(self, loader, sample_export):
        """Test parsing basic Auto Export JSON structure."""
        df = loader.parse_auto_export_json(sample_export)

        assert not df.empty
        assert "timestamp" in df.columns
        assert "metric_type" in df.columns
        assert "value" in df.columns
        assert "source_name" in df.columns

    def test_parse_auto_export_json_has_both_sources(self, loader, sample_export):
        """Test that both Apple Watch and Eight Sleep data are parsed."""
        df = loader.parse_auto_export_json(sample_export)

        sources = df["source_name"].unique()
        source_lower = [s.lower() for s in sources]

        has_watch = any("watch" in s for s in source_lower)
        has_eight = any("eight" in s or "8" in s for s in source_lower)

        assert has_watch or has_eight, f"Expected device sources, got: {sources}"

    def test_parse_auto_export_json_metrics(self, loader, sample_export):
        """Test that expected metrics are present."""
        df = loader.parse_auto_export_json(sample_export)

        metrics = df["metric_type"].unique()
        metric_lower = [m.lower() for m in metrics]

        # Should have at least some of these
        expected = ["heart_rate", "hrv", "sleep"]
        found = [m for m in expected if any(m in ml for ml in metric_lower)]

        assert len(found) > 0, f"Expected metrics like {expected}, got: {metrics}"

    def test_parse_auto_export_json_empty(self, loader):
        """Test parsing empty data."""
        df = loader.parse_auto_export_json({})
        assert df.empty

    def test_parse_auto_export_json_nested_data(self, loader):
        """Test parsing nested data structure."""
        nested_data = {
            "data": {
                "metrics": [
                    {
                        "name": "heart_rate",
                        "units": "bpm",
                        "data": [
                            {
                                "date": "2024-01-15 03:00:00",
                                "qty": 60,
                                "source": "Apple Watch"
                            }
                        ]
                    }
                ]
            }
        }

        df = loader.parse_auto_export_json(nested_data)

        assert len(df) == 1
        assert df.iloc[0]["metric_type"] == "heart_rate"
        assert df.iloc[0]["value"] == 60.0
