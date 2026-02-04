#!/usr/bin/env python3
"""
Import Apple Health Export to create baseline data.

Usage:
    1. Export your Health data: Health app → Profile → Export All Health Data
    2. Unzip the export (creates apple_health_export folder)
    3. Run: python import_health_export.py /path/to/apple_health_export

This creates cache/healthkit_baseline.parquet which the dashboard will automatically use.
"""

import sys
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.healthkit_xml_parser import create_baseline_from_xml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python import_health_export.py ~/Downloads/apple_health_export")
        print("  python import_health_export.py ~/Downloads/apple_health_export/export.xml")
        sys.exit(1)

    export_path = Path(sys.argv[1])

    # Handle zip file
    if export_path.suffix == ".zip":
        print(f"Please unzip {export_path} first, then run with the unzipped folder.")
        sys.exit(1)

    # Find export.xml
    if export_path.is_dir():
        xml_candidates = [
            export_path / "export.xml",
            export_path / "apple_health_export" / "export.xml",
        ]
        xml_path = None
        for candidate in xml_candidates:
            if candidate.exists():
                xml_path = candidate
                break

        if xml_path is None:
            print(f"Could not find export.xml in {export_path}")
            print("Expected locations:")
            for c in xml_candidates:
                print(f"  - {c}")
            sys.exit(1)
    else:
        xml_path = export_path

    if not xml_path.exists():
        print(f"File not found: {xml_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Apple Health Export Import")
    print(f"{'='*60}")
    print(f"Source: {xml_path}")
    print(f"Size: {xml_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nThis may take several minutes for large exports...")
    print(f"{'='*60}\n")

    try:
        output_path = create_baseline_from_xml(str(xml_path))
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Baseline saved to: {output_path}")
        print(f"\nThe dashboard will now automatically merge this baseline")
        print(f"with Auto Export data for complete coverage.")
        print(f"\nRestart the dashboard to use the new baseline:")
        print(f"  pkill -f 'streamlit run' && streamlit run src/dashboard/app.py")
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
