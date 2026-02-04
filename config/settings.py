"""Application configuration."""

import os

# GCP Settings
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "sleep-health-dashboard")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET", "sleep-health-dashboard-healthkit-exports")
GCS_EXPORTS_PREFIX = "exports/"

# Eight Sleep API Settings
# Set these environment variables or update directly (not recommended for passwords)
EIGHT_SLEEP_EMAIL = os.getenv("EIGHT_SLEEP_EMAIL", "")
EIGHT_SLEEP_PASSWORD = os.getenv("EIGHT_SLEEP_PASSWORD", "")
EIGHT_SLEEP_ENABLED = os.getenv("EIGHT_SLEEP_ENABLED", "false").lower() == "true"

# Device source patterns (case-insensitive matching)
APPLE_WATCH_SOURCE_PATTERNS = [
    "apple watch",
    "watch",
]

EIGHT_SLEEP_SOURCE_PATTERNS = [
    "eight sleep",
    "8 sleep",
    "eightsleep",
]

# Night definition
NIGHT_BOUNDARY_HOUR = 18  # 6 PM - sleep after this time belongs to current date
MIN_SLEEP_DURATION_MINUTES = 30
MAX_SLEEP_GAP_MINUTES = 120  # Merge sessions with gaps smaller than this

# Timezone
LOCAL_TIMEZONE = "America/New_York"

# Location for sunrise/sunset calculations (default: Boston)
LATITUDE = float(os.getenv("LATITUDE", "42.3601"))
LONGITUDE = float(os.getenv("LONGITUDE", "-71.0589"))

# Dashboard defaults
DEFAULT_DATE_RANGE_DAYS = 30
AGREEMENT_THRESHOLD_GOOD = 90  # Agreement score above this is "good"
DISCREPANCY_THRESHOLD_MINUTES = 30  # Flag nights with larger differences

# Colors
COLOR_APPLE_WATCH = "#3B82F6"
COLOR_EIGHT_SLEEP = "#EF4444"
COLOR_NEUTRAL = "#6B7280"
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"

# HealthKit metric types
METRIC_SLEEP_ANALYSIS = "HKCategoryTypeIdentifierSleepAnalysis"
METRIC_HEART_RATE = "HKQuantityTypeIdentifierHeartRate"
METRIC_HRV = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
METRIC_RESPIRATORY_RATE = "HKQuantityTypeIdentifierRespiratoryRate"

# Sleep analysis values
SLEEP_VALUE_IN_BED = 0
SLEEP_VALUE_ASLEEP = 1
SLEEP_VALUE_AWAKE = 2
SLEEP_VALUE_CORE = 3
SLEEP_VALUE_DEEP = 4
SLEEP_VALUE_REM = 5
