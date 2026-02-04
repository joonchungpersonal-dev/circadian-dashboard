"""Sleep Dashboard — Sleep Analytics, Circadian Metrics & Trends."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import date, timedelta
import pytz
import logging

import sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    LOCAL_TIMEZONE,
    EIGHT_SLEEP_EMAIL,
    EIGHT_SLEEP_PASSWORD,
    LATITUDE,
    LONGITUDE,
)
import math
from datetime import datetime
from src.ingestion.eight_sleep_api import (
    EightSleepClient,
    get_current_sleep_status,
    get_eight_sleep_data_sync,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Sleep Dashboard",
    page_icon="🌙",
    layout="wide"
)

# Colors - Eight Sleep / Apple style
COLOR_DEEP = "#5E5CE6"      # Indigo for deep sleep
COLOR_REM = "#BF5AF2"       # Purple for REM
COLOR_LIGHT = "#64D2FF"     # Light blue for light sleep
COLOR_AWAKE = "#FF9F0A"     # Orange for awake
COLOR_PRIMARY = "#30D158"   # Green accent
COLOR_HR = "#FF375F"        # Red for heart rate
COLOR_HRV = "#5E5CE6"       # Indigo for HRV
COLOR_SUNRISE = "#FFD60A"   # Yellow for sunrise
COLOR_PROCESS_S = "#FF6B6B" # Red for sleep pressure
COLOR_PROCESS_C = "#4ECDC4" # Teal for circadian


def calculate_sunrise(date_obj: date, latitude: float = LATITUDE, longitude: float = LONGITUDE) -> float:
    """Calculate sunrise time for a given date and location.

    Returns decimal hour in local timezone (e.g., 6.5 = 6:30 AM).
    Uses simplified solar calculation algorithm.
    """
    # Day of year
    n = date_obj.timetuple().tm_yday

    # Solar declination (simplified)
    declination = 23.45 * math.sin(math.radians((360 / 365) * (n - 81)))

    # Hour angle at sunrise
    lat_rad = math.radians(latitude)
    dec_rad = math.radians(declination)

    # Clamp for polar regions
    cos_hour_angle = -math.tan(lat_rad) * math.tan(dec_rad)
    cos_hour_angle = max(-1, min(1, cos_hour_angle))

    hour_angle = math.degrees(math.acos(cos_hour_angle))

    # Solar noon (approximation based on longitude)
    # Standard timezone offset for longitude
    tz_offset = round(longitude / 15)  # Approximate timezone
    solar_noon = 12 - (longitude - tz_offset * 15) / 15

    # Sunrise time (local solar time)
    sunrise_solar = solar_noon - hour_angle / 15

    # Adjust for actual timezone (rough approximation)
    # For more accuracy, would need proper timezone handling
    local_tz = pytz.timezone(LOCAL_TIMEZONE)

    # Get UTC offset for the date
    dt = datetime(date_obj.year, date_obj.month, date_obj.day, 12, 0)
    utc_offset = local_tz.utcoffset(dt).total_seconds() / 3600

    # Longitude-based offset from timezone center
    tz_center_longitude = utc_offset * 15
    longitude_correction = (longitude - tz_center_longitude) / 15

    sunrise_local = sunrise_solar - longitude_correction

    # Ensure reasonable bounds (4 AM to 9 AM typically)
    return max(4.0, min(9.0, sunrise_local))


@st.cache_data(ttl=300, show_spinner=False)
def load_eight_sleep_status(email: str, password: str) -> dict:
    """Load current Eight Sleep status."""
    if not email or not password:
        return {}
    try:
        return get_current_sleep_status(email, password, LOCAL_TIMEZONE)
    except Exception as e:
        logger.error(f"Eight Sleep API error: {e}")
        return {"error": str(e)}


@st.cache_data(ttl=600, show_spinner=False)
def load_eight_sleep_data(email: str, password: str, days: int) -> pd.DataFrame:
    """Load Eight Sleep historical data."""
    if not email or not password:
        return pd.DataFrame()
    try:
        return get_eight_sleep_data_sync(email, password, days=days)
    except Exception as e:
        logger.error(f"Eight Sleep data error: {e}")
        return pd.DataFrame()


def create_sleep_timeline(df: pd.DataFrame) -> go.Figure:
    """Create horizontal sleep bars like Eight Sleep app."""
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="No sleep data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#8E8E93")
        )
        fig.update_layout(
            plot_bgcolor="#1C1C1E",
            paper_bgcolor="#1C1C1E",
            height=200
        )
        return fig

    local_tz = pytz.timezone(LOCAL_TIMEZONE)
    df = df.sort_values("night_date", ascending=False)

    def to_decimal_hour(ts):
        if pd.isna(ts):
            return None
        if hasattr(ts, 'astimezone'):
            local_ts = ts.astimezone(local_tz)
        else:
            local_ts = ts
        hour = local_ts.hour + local_ts.minute / 60
        if hour < 12:
            hour += 24
        return hour

    dates = []
    starts = []
    durations = []
    hover_texts = []

    for _, row in df.iterrows():
        night = row["night_date"]
        sleep_start = to_decimal_hour(row.get("sleep_start"))
        sleep_end = to_decimal_hour(row.get("sleep_end"))

        if sleep_start is None or sleep_end is None:
            # Use session_start and duration if available
            session_start = row.get("session_start")
            duration_hours = row.get("duration_hours", 0)
            if session_start is not None and duration_hours > 0:
                sleep_start = to_decimal_hour(session_start)
                sleep_end = sleep_start + duration_hours if sleep_start else None

        if sleep_start is None or sleep_end is None:
            continue

        duration = row.get("duration_hours", 0)
        deep = row.get("deep_minutes", 0) or 0
        rem = row.get("rem_minutes", 0) or 0
        light = row.get("light_minutes", 0) or 0
        total_mins = duration * 60 if duration > 0 else 1
        deep_pct = (deep / total_mins) * 100
        rem_pct = (rem / total_mins) * 100

        date_label = night.strftime("%a %-m/%-d")
        dates.append(date_label)
        starts.append(sleep_start)
        durations.append(sleep_end - sleep_start)

        hover_texts.append(
            f"<b>{night.strftime('%A, %b %-d')}</b><br>"
            f"<b>{duration:.1f} hours</b><br>"
            f"Deep: {deep:.0f}m ({deep_pct:.0f}%) | REM: {rem:.0f}m ({rem_pct:.0f}%) | Light: {light:.0f}m"
        )

    fig.add_trace(go.Bar(
        y=dates,
        x=durations,
        base=starts,
        orientation="h",
        marker=dict(
            color=COLOR_PRIMARY,
            line=dict(width=0),
            cornerradius=6,
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        showlegend=False,
    ))

    # Add sunrise markers for each night
    sunrise_x = []
    sunrise_y = []
    sunrise_texts = []

    for _, row in df.iterrows():
        night = row["night_date"]
        # Sunrise is on the morning after the night date
        wake_date = night + timedelta(days=1)
        sunrise_hour = calculate_sunrise(wake_date)
        # Convert to same scale (hours after noon previous day, so 6 AM = 30)
        sunrise_decimal = sunrise_hour + 24  # e.g., 6.5 AM becomes 30.5

        date_label = night.strftime("%a %-m/%-d")
        sunrise_x.append(sunrise_decimal)
        sunrise_y.append(date_label)

        sunrise_time = f"{int(sunrise_hour)}:{int((sunrise_hour % 1) * 60):02d} AM"
        sunrise_texts.append(f"Sunrise: {sunrise_time}")

    # Add sunrise markers (subtle diamond)
    fig.add_trace(go.Scatter(
        x=sunrise_x,
        y=sunrise_y,
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=8,
            color="#FFA726",
            opacity=0.8,
            line=dict(width=1, color="#FFB74D"),
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=sunrise_texts,
        showlegend=False,
        name="Sunrise",
    ))

    tick_vals = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    tick_text = ["9p", "10p", "11p", "12a", "1a", "2a", "3a", "4a", "5a", "6a", "7a", "8a", "9a", "10a", "11a"]

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10, color="#8E8E93"),
            range=[20.5, 35.5],
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            showline=False,
            side="top",
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#FFFFFF"),
            categoryorder="array",
            categoryarray=dates,
            gridcolor="rgba(255,255,255,0.03)",
            zeroline=False,
            showline=False,
        ),
        barmode="overlay",
        bargap=0.35,
        height=max(280, len(dates) * 32 + 60),
        margin=dict(l=75, r=15, t=30, b=15),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        hoverlabel=dict(
            bgcolor="#2C2C2E",
            bordercolor="#3A3A3C",
            font=dict(size=12, color="#FFFFFF"),
        ),
    )

    fig.add_vline(x=24, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))

    return fig


def create_hypnogram(row: pd.Series) -> go.Figure:
    """Create hypnogram for a single night's sleep."""
    fig = go.Figure()

    stages = row.get("stages", [])
    if not stages:
        fig.add_annotation(
            text="No stage data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#8E8E93")
        )
        fig.update_layout(
            plot_bgcolor="#1C1C1E",
            paper_bgcolor="#1C1C1E",
            height=200
        )
        return fig

    # Stage to y-value mapping (higher = lighter sleep)
    stage_map = {"deep": 0, "rem": 1, "light": 2, "awake": 3}
    stage_colors = {"deep": COLOR_DEEP, "rem": COLOR_REM, "light": COLOR_LIGHT, "awake": COLOR_AWAKE}

    # Find first actual sleep stage (skip initial awake/in-bed time)
    sleep_start_offset = 0
    for stage in stages:
        if stage["stage"] in ("deep", "light", "rem"):
            sleep_start_offset = stage["start_min"]
            break

    # Filter out trailing awake period and adjust times
    filtered_stages = []
    for stage in stages:
        adjusted_start = stage["start_min"] - sleep_start_offset
        adjusted_end = stage["end_min"] - sleep_start_offset
        # Skip stages that are entirely before sleep started
        if adjusted_end <= 0:
            continue
        # Clip start to 0 if it's negative
        adjusted_start = max(0, adjusted_start)
        filtered_stages.append({
            **stage,
            "start_min": adjusted_start,
            "end_min": adjusted_end,
        })

    # Remove trailing awake period (after last sleep stage)
    if filtered_stages and filtered_stages[-1]["stage"] == "awake":
        filtered_stages = filtered_stages[:-1]

    stages = filtered_stages

    # Build step chart data for connecting line
    line_x = []
    line_y = []

    for stage in stages:
        stage_type = stage["stage"]
        if stage_type not in stage_map:
            continue
        y_val = stage_map[stage_type]
        line_x.extend([stage["start_min"], stage["end_min"]])
        line_y.extend([y_val, y_val])

    # Add connecting step line (background)
    fig.add_trace(go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.4)", width=2, shape="hv"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Create colored segments for each stage (foreground)
    for stage in stages:
        stage_type = stage["stage"]
        if stage_type not in stage_map:
            continue

        fig.add_trace(go.Scatter(
            x=[stage["start_min"], stage["end_min"]],
            y=[stage_map[stage_type], stage_map[stage_type]],
            mode="lines",
            line=dict(color=stage_colors.get(stage_type, "#888"), width=6),
            hovertemplate=f"{stage_type.title()}: {stage['duration_min']:.0f}m<extra></extra>",
            showlegend=False,
        ))

    # Calculate hours for x-axis
    max_mins = max(s["end_min"] for s in stages) if stages else 480
    tick_vals = list(range(0, int(max_mins) + 60, 60))
    tick_text = [f"{m//60}h" for m in tick_vals]

    fig.update_layout(
        xaxis=dict(
            title="Time Asleep",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10, color="#8E8E93"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1, 2, 3],
            ticktext=["Deep", "REM", "Light", "Awake"],
            tickfont=dict(size=11, color="#FFFFFF"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            range=[-0.5, 3.5],
        ),
        height=220,
        margin=dict(l=60, r=20, t=30, b=40),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
    )

    return fig


def compute_sleep_variability(df: pd.DataFrame) -> dict:
    """Compute standard deviations for sleep timing and duration.

    Only uses main sleep sessions (longest per night, started 6PM-6AM).
    """
    local_tz = pytz.timezone(LOCAL_TIMEZONE)

    def is_main_sleep(row):
        """Check if session is likely main sleep (started 6 PM - 6 AM)."""
        ts = row.get("session_start")
        if pd.isna(ts):
            return False
        if hasattr(ts, 'astimezone'):
            local_ts = ts.astimezone(local_tz)
        else:
            local_ts = ts
        hour = local_ts.hour
        # Main sleep typically starts between 6 PM (18) and 6 AM (6)
        return hour >= 18 or hour < 6

    def time_to_minutes(ts):
        """Convert timestamp to minutes from midnight (adjusted for overnight)."""
        if pd.isna(ts):
            return None
        if hasattr(ts, 'astimezone'):
            local_ts = ts.astimezone(local_tz)
        else:
            local_ts = ts
        mins = local_ts.hour * 60 + local_ts.minute
        # Adjust for overnight (times after midnight but before 6 AM become 24:xx - 30:xx)
        if local_ts.hour < 6:
            mins += 1440  # Add 24 hours
        return mins

    def compute_sd(values):
        """Compute standard deviation, return None if insufficient data."""
        valid = [v for v in values if v is not None]
        if len(valid) < 2:
            return None
        return np.std(valid)

    # Filter to main sleep sessions only
    main_df = df[df.apply(is_main_sleep, axis=1)].copy()

    # If multiple sessions per night, keep the longest
    if not main_df.empty:
        main_df = main_df.loc[main_df.groupby("night_date")["duration_hours"].idxmax()]

    if main_df.empty:
        return {"onset_sd": None, "wake_sd": None, "midpoint_sd": None, "duration_sd": None}

    # Compute onset times (session_start)
    onset_mins = [time_to_minutes(ts) for ts in main_df["session_start"]]
    onset_sd = compute_sd(onset_mins)

    # Compute wake times (session_start + duration)
    wake_mins = []
    for _, row in main_df.iterrows():
        start_ts = row.get("session_start")
        duration_hrs = row.get("duration_hours", 0)
        if pd.notna(start_ts) and duration_hrs > 0:
            start_min = time_to_minutes(start_ts)
            if start_min is not None:
                wake_mins.append(start_min + duration_hrs * 60)
    wake_sd = compute_sd(wake_mins)

    # Compute midpoint
    midpoint_mins = []
    for onset, wake in zip(onset_mins, wake_mins):
        if onset is not None and wake is not None:
            midpoint_mins.append((onset + wake) / 2)
    midpoint_sd = compute_sd(midpoint_mins)

    # Duration SD (in minutes) - use filtered main_df
    duration_mins = [d * 60 for d in main_df["duration_hours"].dropna()]
    duration_sd = compute_sd(duration_mins)

    return {
        "onset_sd": onset_sd,
        "wake_sd": wake_sd,
        "midpoint_sd": midpoint_sd,
        "duration_sd": duration_sd,
    }


def compute_circadian_metrics(df: pd.DataFrame) -> dict:
    """Compute circadian rhythm metrics: sleep midpoint and social jetlag.

    Sleep Midpoint: Clock time at the middle of the sleep period.
    Social Jetlag: Difference in sleep midpoint between work days (Mon-Fri) and free days (Sat-Sun).

    Returns dict with:
        - avg_midpoint_time: Average sleep midpoint as time string (e.g., "3:45 AM")
        - avg_midpoint_mins: Average midpoint in minutes from midnight (adjusted for overnight)
        - weekday_midpoint_mins: Average midpoint on Mon-Fri
        - weekend_midpoint_mins: Average midpoint on Sat-Sun
        - social_jetlag_mins: Difference (weekend - weekday midpoint)
        - midpoints_by_night: List of (night_date, midpoint_mins) for plotting
    """
    local_tz = pytz.timezone(LOCAL_TIMEZONE)

    def is_main_sleep(row):
        """Check if session is likely main sleep (started 6 PM - 6 AM)."""
        ts = row.get("session_start")
        if pd.isna(ts):
            return False
        if hasattr(ts, 'astimezone'):
            local_ts = ts.astimezone(local_tz)
        else:
            local_ts = ts
        hour = local_ts.hour
        return hour >= 18 or hour < 6

    def time_to_minutes(ts):
        """Convert timestamp to minutes from midnight (adjusted for overnight)."""
        if pd.isna(ts):
            return None
        if hasattr(ts, 'astimezone'):
            local_ts = ts.astimezone(local_tz)
        else:
            local_ts = ts
        mins = local_ts.hour * 60 + local_ts.minute
        if local_ts.hour < 6:
            mins += 1440
        return mins

    def mins_to_time_str(mins):
        """Convert minutes to readable time string."""
        if mins is None:
            return "N/A"
        # Normalize to 0-1440 range for display
        display_mins = mins % 1440
        hours = int(display_mins // 60)
        minutes = int(display_mins % 60)
        period = "AM" if hours < 12 else "PM"
        display_hour = hours % 12
        if display_hour == 0:
            display_hour = 12
        return f"{display_hour}:{minutes:02d} {period}"

    # Filter to main sleep sessions
    main_df = df[df.apply(is_main_sleep, axis=1)].copy()
    if not main_df.empty:
        main_df = main_df.loc[main_df.groupby("night_date")["duration_hours"].idxmax()]

    if main_df.empty:
        return {
            "avg_midpoint_time": "N/A",
            "avg_midpoint_mins": None,
            "weekday_midpoint_mins": None,
            "weekend_midpoint_mins": None,
            "social_jetlag_mins": None,
            "midpoints_by_night": [],
        }

    # Compute midpoint for each night
    midpoints = []
    for _, row in main_df.iterrows():
        start_ts = row.get("session_start")
        duration_hrs = row.get("duration_hours", 0)
        night = row.get("night_date")

        if pd.isna(start_ts) or duration_hrs <= 0:
            continue

        onset_mins = time_to_minutes(start_ts)
        if onset_mins is None:
            continue

        wake_mins = onset_mins + duration_hrs * 60
        midpoint = (onset_mins + wake_mins) / 2

        # Determine if this is a weekday or weekend night
        # night_date is the date the sleep started (e.g., Friday night -> Saturday sleep)
        # For social jetlag, we care about the wake-up day
        wake_day = (night + timedelta(days=1)).weekday()  # 0=Mon, 6=Sun
        is_free_day = wake_day in (5, 6)  # Saturday or Sunday wake-up

        midpoints.append({
            "night_date": night,
            "midpoint_mins": midpoint,
            "is_free_day": is_free_day,
        })

    if not midpoints:
        return {
            "avg_midpoint_time": "N/A",
            "avg_midpoint_mins": None,
            "weekday_midpoint_mins": None,
            "weekend_midpoint_mins": None,
            "social_jetlag_mins": None,
            "midpoints_by_night": [],
        }

    # Calculate averages
    all_midpoints = [m["midpoint_mins"] for m in midpoints]
    avg_midpoint = np.mean(all_midpoints) if all_midpoints else None

    weekday_midpoints = [m["midpoint_mins"] for m in midpoints if not m["is_free_day"]]
    weekend_midpoints = [m["midpoint_mins"] for m in midpoints if m["is_free_day"]]

    weekday_avg = np.mean(weekday_midpoints) if weekday_midpoints else None
    weekend_avg = np.mean(weekend_midpoints) if weekend_midpoints else None

    # Social jetlag = weekend midpoint - weekday midpoint (positive = later on weekends)
    social_jetlag = None
    if weekday_avg is not None and weekend_avg is not None:
        social_jetlag = weekend_avg - weekday_avg

    return {
        "avg_midpoint_time": mins_to_time_str(avg_midpoint),
        "avg_midpoint_mins": avg_midpoint,
        "weekday_midpoint_time": mins_to_time_str(weekday_avg),
        "weekday_midpoint_mins": weekday_avg,
        "weekend_midpoint_time": mins_to_time_str(weekend_avg),
        "weekend_midpoint_mins": weekend_avg,
        "social_jetlag_mins": social_jetlag,
        "midpoints_by_night": [(m["night_date"], m["midpoint_mins"]) for m in midpoints],
    }


def create_midpoint_chart(midpoints_by_night: list) -> go.Figure:
    """Create sleep midpoint trend chart with dashed line connecting nights."""
    fig = go.Figure()

    if not midpoints_by_night:
        fig.add_annotation(
            text="No midpoint data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#8E8E93")
        )
        fig.update_layout(
            plot_bgcolor="#1C1C1E",
            paper_bgcolor="#1C1C1E",
            height=250
        )
        return fig

    # Sort by date ascending for proper line connection
    sorted_data = sorted(midpoints_by_night, key=lambda x: x[0])
    dates = [d[0] for d in sorted_data]
    midpoints = [d[1] for d in sorted_data]

    # Add dashed line connecting midpoints
    fig.add_trace(go.Scatter(
        x=dates,
        y=midpoints,
        mode="lines+markers",
        line=dict(color="#AF52DE", width=2, dash="dash"),
        marker=dict(color="#AF52DE", size=8),
        name="Sleep Midpoint",
        hovertemplate="<b>%{x|%a %b %d}</b><br>Midpoint: %{customdata}<extra></extra>",
        customdata=[_mins_to_hover_time(m) for m in midpoints],
    ))

    # Add horizontal line for average
    avg_midpoint = np.mean(midpoints)
    fig.add_hline(
        y=avg_midpoint,
        line=dict(color="rgba(175, 82, 222, 0.4)", width=1, dash="dot"),
        annotation_text=f"Avg: {_mins_to_hover_time(avg_midpoint)}",
        annotation_position="right",
        annotation_font=dict(color="#AF52DE", size=10),
    )

    # Y-axis: show times in 30-min increments
    min_y = min(midpoints) - 30
    max_y = max(midpoints) + 30
    tick_vals = list(range(int(min_y // 30) * 30, int(max_y // 30 + 2) * 30, 60))
    tick_text = [_mins_to_hover_time(m) for m in tick_vals]

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=10, color="#8E8E93"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10, color="#8E8E93"),
            gridcolor="rgba(255,255,255,0.06)",
            zeroline=False,
            title="Sleep Midpoint",
        ),
        height=250,
        margin=dict(l=70, r=20, t=30, b=30),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#2C2C2E",
            bordercolor="#3A3A3C",
            font=dict(size=12, color="#FFFFFF"),
        ),
    )

    return fig


def _mins_to_hover_time(mins):
    """Convert minutes to readable time for hover/display."""
    if mins is None:
        return "N/A"
    display_mins = mins % 1440
    hours = int(display_mins // 60)
    minutes = int(display_mins % 60)
    period = "AM" if hours < 12 else "PM"
    display_hour = hours % 12
    if display_hour == 0:
        display_hour = 12
    return f"{display_hour}:{minutes:02d} {period}"


def create_two_process_model(df: pd.DataFrame) -> go.Figure:
    """Create Two-Process Model visualization (Borbély model).

    Process S (Homeostatic): Sleep pressure builds during wake, dissipates during sleep
    Process C (Circadian): ~24hr sinusoidal rhythm from SCN

    Shows how sleep propensity emerges from the interaction of these two processes.
    """
    fig = go.Figure()

    local_tz = pytz.timezone(LOCAL_TIMEZONE)

    # Get average sleep/wake times from data
    def get_avg_times(df):
        onset_hours = []
        wake_hours = []

        for _, row in df.iterrows():
            ts = row.get("session_start")
            duration = row.get("duration_hours", 0)
            if pd.isna(ts) or duration <= 0:
                continue

            if hasattr(ts, 'astimezone'):
                local_ts = ts.astimezone(local_tz)
            else:
                local_ts = ts

            onset_hour = local_ts.hour + local_ts.minute / 60
            # Adjust for overnight (e.g., 11 PM = 23, 1 AM = 25)
            if onset_hour < 12:
                onset_hour += 24
            onset_hours.append(onset_hour)
            wake_hours.append(onset_hour + duration)

        if not onset_hours:
            return 23.0, 7.0  # Default: 11 PM - 7 AM
        return np.mean(onset_hours), np.mean(wake_hours)

    avg_onset, avg_wake = get_avg_times(df)

    # Normalize times: start from 6 PM (18:00) as hour 0
    # This gives us a nice view of evening -> night -> morning
    base_hour = 18  # 6 PM

    def normalize_hour(h):
        """Convert clock hour to hours since 6 PM."""
        if h >= base_hour:
            return h - base_hour
        else:
            return h + 24 - base_hour

    sleep_onset_norm = normalize_hour(avg_onset % 24)
    sleep_duration = avg_wake - avg_onset
    wake_time_norm = sleep_onset_norm + sleep_duration

    # Generate time points for 36 hours (6 PM to 6 AM next day + extra)
    t = np.linspace(0, 36, 500)

    # === PROCESS S (Homeostatic Sleep Pressure) ===
    # Rises exponentially during wake, falls exponentially during sleep
    # S(t) during wake: S_max - (S_max - S_min) * exp(-t/tau_rise)
    # S(t) during sleep: S_min + (S_prev - S_min) * exp(-t/tau_fall)

    S_max = 1.0  # Upper asymptote
    S_min = 0.15  # Lower asymptote
    tau_rise = 18.2  # Time constant for rise (hours) - typical value
    tau_fall = 4.2   # Time constant for fall (hours) - typical value

    process_s = np.zeros_like(t)

    # Start at moderate sleep pressure (end of typical day)
    S_current = 0.7

    for i, hour in enumerate(t):
        # Determine if asleep or awake
        # First sleep period
        if sleep_onset_norm <= hour < wake_time_norm:
            # Sleeping - S decreases
            time_asleep = hour - sleep_onset_norm
            S_at_onset = S_current if i == 0 else process_s[i-1]
            if time_asleep == 0:
                process_s[i] = S_at_onset
            else:
                # Exponential decay toward S_min
                hours_since_onset = hour - sleep_onset_norm
                process_s[i] = S_min + (0.95 - S_min) * np.exp(-hours_since_onset / tau_fall)
        # Second sleep period (next night)
        elif sleep_onset_norm + 24 <= hour < wake_time_norm + 24:
            time_asleep = hour - (sleep_onset_norm + 24)
            hours_since_onset = hour - (sleep_onset_norm + 24)
            process_s[i] = S_min + (0.95 - S_min) * np.exp(-hours_since_onset / tau_fall)
        else:
            # Awake - S increases
            if hour < sleep_onset_norm:
                # Before first sleep
                hours_awake = hour + (24 - wake_time_norm + sleep_onset_norm) % 24
                # Simplified: assume woke at avg wake time
                hours_awake = hour + 6  # Assume 6 hours since wake before 6 PM
            elif hour >= wake_time_norm and hour < sleep_onset_norm + 24:
                # After first sleep, before second
                hours_awake = hour - wake_time_norm
            else:
                # After second sleep
                hours_awake = hour - (wake_time_norm + 24)

            # Exponential rise toward S_max
            S_at_wake = S_min + 0.05
            process_s[i] = S_max - (S_max - S_at_wake) * np.exp(-hours_awake / tau_rise)

    # === PROCESS C (Circadian Alerting Signal) ===
    # Sinusoidal with period ~24h, peaks in evening (~21:00), nadir early morning (~05:00)
    # Higher values = more alerting (opposing sleep)

    # Peak alerting at ~9 PM (21:00) = 3 hours after our base (6 PM)
    peak_hour_norm = 3  # 9 PM in normalized time
    C_amplitude = 0.35
    C_baseline = 0.5

    # Cosine wave: peaks at peak_hour_norm
    process_c = C_baseline + C_amplitude * np.cos(2 * np.pi * (t - peak_hour_norm) / 24)

    # === SLEEP PROPENSITY ===
    # When S exceeds C, sleep is likely
    # Sleep propensity = S - C (positive = sleepy)

    # Add traces
    fig.add_trace(go.Scatter(
        x=t,
        y=process_s,
        name="Process S (Sleep Pressure)",
        line=dict(color=COLOR_PROCESS_S, width=3),
        hovertemplate="Hour %{x:.1f}<br>Sleep Pressure: %{y:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=t,
        y=process_c,
        name="Process C (Circadian)",
        line=dict(color=COLOR_PROCESS_C, width=3),
        hovertemplate="Hour %{x:.1f}<br>Circadian Alert: %{y:.2f}<extra></extra>",
    ))

    # Add shaded sleep periods
    fig.add_vrect(
        x0=sleep_onset_norm, x1=wake_time_norm,
        fillcolor="rgba(94, 92, 230, 0.2)",
        line_width=0,
        annotation_text="Sleep",
        annotation_position="top",
        annotation_font=dict(color="#8E8E93", size=10),
    )

    if wake_time_norm + 24 <= 36:
        fig.add_vrect(
            x0=sleep_onset_norm + 24, x1=min(wake_time_norm + 24, 36),
            fillcolor="rgba(94, 92, 230, 0.2)",
            line_width=0,
        )

    # Add "sleep gate" annotation where S first exceeds C
    for i in range(1, len(t)):
        if process_s[i] > process_c[i] and process_s[i-1] <= process_c[i-1]:
            fig.add_annotation(
                x=t[i], y=process_s[i],
                text="Sleep Gate",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#8E8E93",
                font=dict(size=10, color="#8E8E93"),
                ax=30, ay=-30,
            )
            break

    # X-axis labels (convert back to clock time)
    tick_vals = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36]
    tick_text = ["6p", "9p", "12a", "3a", "6a", "9a", "12p", "3p", "6p", "9p", "12a", "3a", "6a"]

    fig.update_layout(
        xaxis=dict(
            title="Time",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10, color="#8E8E93"),
            gridcolor="rgba(255,255,255,0.06)",
            range=[0, 36],
        ),
        yaxis=dict(
            title="Level",
            tickfont=dict(size=10, color="#8E8E93"),
            gridcolor="rgba(255,255,255,0.06)",
            range=[0, 1.1],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        height=300,
        margin=dict(l=50, r=20, t=50, b=40),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
        hoverlabel=dict(
            bgcolor="#2C2C2E",
            bordercolor="#3A3A3C",
            font=dict(size=12, color="#FFFFFF"),
        ),
    )

    return fig


def create_hr_chart(df: pd.DataFrame) -> go.Figure:
    """Create heart rate trend chart."""
    fig = go.Figure()

    if df.empty:
        return fig

    df = df.sort_values("night_date", ascending=True)
    df = df.dropna(subset=["heart_rate_avg"])

    if df.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["heart_rate_avg"],
        name="Heart Rate",
        line=dict(color=COLOR_HR, width=2),
        mode="lines+markers",
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(255, 55, 95, 0.1)",
    ))

    # Add 7-day rolling average
    if len(df) >= 7:
        df["hr_rolling"] = df["heart_rate_avg"].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["night_date"],
            y=df["hr_rolling"],
            name="7-day Avg",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            mode="lines",
        ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title="bpm",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=250,
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
        xaxis=dict(tickfont=dict(color="#8E8E93")),
        yaxis=dict(tickfont=dict(color="#8E8E93"), gridcolor="rgba(255,255,255,0.06)"),
    )

    return fig


def create_hrv_chart(df: pd.DataFrame) -> go.Figure:
    """Create HRV trend chart."""
    fig = go.Figure()

    if df.empty:
        return fig

    df = df.sort_values("night_date", ascending=True)
    df = df.dropna(subset=["hrv_avg"])

    if df.empty:
        return fig

    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hrv_avg"],
        name="HRV",
        line=dict(color=COLOR_HRV, width=2),
        mode="lines+markers",
        marker=dict(size=5),
        fill="tozeroy",
        fillcolor="rgba(94, 92, 230, 0.1)",
    ))

    # Add 7-day rolling average
    if len(df) >= 7:
        df["hrv_rolling"] = df["hrv_avg"].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["night_date"],
            y=df["hrv_rolling"],
            name="7-day Avg",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            mode="lines",
        ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title="ms",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=250,
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
        xaxis=dict(tickfont=dict(color="#8E8E93")),
        yaxis=dict(tickfont=dict(color="#8E8E93"), gridcolor="rgba(255,255,255,0.06)"),
    )

    return fig


def create_correlation_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                               x_label: str, y_label: str, color: str) -> go.Figure:
    """Create a scatter plot with correlation line and coefficient."""
    fig = go.Figure()

    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return fig

    # Drop rows with missing values
    plot_df = df[[x_col, y_col]].dropna()
    if len(plot_df) < 3:
        return fig

    x = plot_df[x_col]
    y = plot_df[y_col]

    # Calculate correlation
    corr = x.corr(y)

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(color=color, size=10, opacity=0.7),
        hovertemplate=f"{x_label}: %{{x:.1f}}<br>{y_label}: %{{y:.1f}}<extra></extra>",
        showlegend=False,
    ))

    # Add trend line
    if len(x) >= 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 50)
        fig.add_trace(go.Scatter(
            x=x_line,
            y=p(x_line),
            mode="lines",
            line=dict(color="rgba(255,255,255,0.5)", width=2, dash="dash"),
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text=f"r = {corr:.2f}",
            font=dict(size=14, color="#FFFFFF"),
            x=0.5,
            xanchor="center",
        ),
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=250,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
        xaxis=dict(tickfont=dict(color="#8E8E93"), gridcolor="rgba(255,255,255,0.06)"),
        yaxis=dict(tickfont=dict(color="#8E8E93"), gridcolor="rgba(255,255,255,0.06)"),
    )

    return fig


def get_llm_analysis(sleep_data: dict) -> str:
    """Get sleep analysis from local Qwen model."""
    try:
        prompt = f"""You are a sleep health expert. Analyze this sleep data and provide:
1. Overall sleep quality assessment
2. Key patterns or concerns
3. 2-3 specific recommendations

Data:
- Nights tracked: {sleep_data.get('total_nights', 0)}
- Avg duration: {sleep_data.get('avg_duration', 'N/A')} hours
- Avg deep sleep: {sleep_data.get('avg_deep', 'N/A')} minutes
- Avg REM: {sleep_data.get('avg_rem', 'N/A')} minutes
- Avg HR: {sleep_data.get('avg_hr', 'N/A')} bpm
- Avg HRV: {sleep_data.get('avg_hrv', 'N/A')} ms

Be concise and actionable."""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 500}
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get("response", "No response")
        return f"Error: {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "Ollama not running. Start with: `ollama serve`"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    """Main dashboard."""

    # Sidebar
    with st.sidebar:
        st.title("Sleep Dashboard")
        st.caption("Sleep Analytics & Circadian Metrics")
        st.divider()

        # Date range
        st.subheader("Date Range")
        days = st.selectbox(
            "Show data for",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )

        st.divider()

        # Credentials
        email = EIGHT_SLEEP_EMAIL
        password = EIGHT_SLEEP_PASSWORD

        if not email or not password:
            st.warning("Configure credentials")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            st.caption("Or set EIGHT_SLEEP_EMAIL and EIGHT_SLEEP_PASSWORD env vars")
        else:
            st.success("Connected")
            st.caption(f"{email[:3]}...@{email.split('@')[1]}")

        st.divider()

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Check credentials
    if not email or not password:
        st.warning("Please configure your Eight Sleep credentials in the sidebar.")
        return

    # Load data
    with st.spinner("Loading Eight Sleep data..."):
        df = load_eight_sleep_data(email, password, days)

    if df.empty:
        st.error("Could not load data. Check your credentials and try again.")
        return

    # Filter out nights with insufficient data (aggregate first, then filter)
    night_totals = df.groupby("night_date")["duration_hours"].sum()
    valid_nights = night_totals[night_totals >= 4.0].index
    df = df[df["night_date"].isin(valid_nights)]

    # === DASHBOARD ===

    # Aggregate sessions by night (sum durations for nights with multiple sessions)
    daily_df = df.groupby("night_date").agg({
        "duration_hours": "sum",
        "deep_minutes": "sum",
        "rem_minutes": "sum",
        "light_minutes": "sum",
        "awake_minutes": "sum",
        "heart_rate_avg": "mean",
        "hrv_avg": "mean",
        "session_start": "first",  # First session of the night
        "sleep_end": "last",  # Last session end
    }).reset_index()
    daily_df = daily_df.sort_values("night_date", ascending=False)

    # Filter out incomplete nights (less than 4 hours total sleep)
    daily_df = daily_df[daily_df["duration_hours"] >= 4.0]

    # Get main sleep session per night for hypnogram (longest session)
    main_sessions = df.loc[df.groupby("night_date")["duration_hours"].idxmax()]

    local_tz = pytz.timezone(LOCAL_TIMEZONE)

    # Last Night Summary
    st.header("Last Night")

    last_night = daily_df.iloc[0]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        duration = last_night.get("duration_hours", 0)
        st.metric("Sleep Duration", f"{duration:.1f}h")

    with col2:
        deep = last_night.get("deep_minutes", 0) or 0
        total_sleep = duration * 60 if duration > 0 else 1
        deep_pct = (deep / total_sleep) * 100
        st.metric("Deep Sleep", f"{deep:.0f}m ({deep_pct:.0f}%)")

    with col3:
        rem = last_night.get("rem_minutes", 0) or 0
        rem_pct = (rem / total_sleep) * 100
        st.metric("REM Sleep", f"{rem:.0f}m ({rem_pct:.0f}%)")

    with col4:
        hr = last_night.get("heart_rate_avg")
        st.metric("Avg Heart Rate", f"{hr:.0f} bpm" if hr and not pd.isna(hr) else "N/A")

    with col5:
        hrv = last_night.get("hrv_avg")
        st.metric("HRV", f"{hrv:.0f} ms" if hrv and not pd.isna(hrv) else "N/A")

    st.divider()

    # Averages (from daily aggregates)
    st.header(f"{days}-Day Averages")

    num_nights = len(daily_df)
    avg_dur = daily_df["duration_hours"].mean()
    avg_deep = daily_df["deep_minutes"].mean()
    avg_rem = daily_df["rem_minutes"].mean()
    avg_total = avg_dur * 60 if avg_dur > 0 else 1
    avg_deep_pct = (avg_deep / avg_total) * 100
    avg_rem_pct = (avg_rem / avg_total) * 100

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Avg Sleep", f"{avg_dur:.1f}h")

    with col2:
        st.metric("Avg Deep", f"{avg_deep:.0f}m ({avg_deep_pct:.0f}%)")

    with col3:
        st.metric("Avg REM", f"{avg_rem:.0f}m ({avg_rem_pct:.0f}%)")

    with col4:
        avg_hr = daily_df["heart_rate_avg"].dropna().mean()
        st.metric("Avg HR", f"{avg_hr:.0f} bpm" if not pd.isna(avg_hr) else "N/A")

    with col5:
        avg_hrv = daily_df["hrv_avg"].dropna().mean()
        st.metric("Avg HRV", f"{avg_hrv:.0f} ms" if not pd.isna(avg_hrv) else "N/A")

    with col6:
        st.metric("Nights Tracked", num_nights)

    st.divider()

    # Sleep Timeline (last 14 days)
    st.header("Sleep Timeline")
    timeline_df = daily_df.head(14).copy()
    # Need session_start for timeline - use raw df
    timeline_raw = df[df["night_date"].isin(timeline_df["night_date"])]
    fig = create_sleep_timeline(timeline_raw)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Hypnogram and Sleep Variability
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Last Night Hypnogram")
        # Get the main session for last night
        last_main = main_sessions.iloc[0] if not main_sessions.empty else None
        if last_main is not None and "stages" in last_main and last_main["stages"]:
            fig = create_hypnogram(last_main)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hypnogram data available")

    with col2:
        st.subheader("Sleep Schedule Variability")
        variability = compute_sleep_variability(df)

        def format_sd(value, label):
            """Format SD value with green/red indicator."""
            if value is None:
                return f"{label}: N/A"
            status = "🟢" if value < 60 else "🔴"
            return f"{status} {label}: {value:.0f} min"

        m1, m2 = st.columns(2)
        with m1:
            onset_sd = variability.get("onset_sd")
            status = "🟢 Good" if onset_sd and onset_sd < 60 else "🔴 High" if onset_sd else ""
            st.metric("Onset SD", f"{onset_sd:.0f} min" if onset_sd else "N/A", delta=status, delta_color="off")

            midpoint_sd = variability.get("midpoint_sd")
            status = "🟢 Good" if midpoint_sd and midpoint_sd < 60 else "🔴 High" if midpoint_sd else ""
            st.metric("Midpoint SD", f"{midpoint_sd:.0f} min" if midpoint_sd else "N/A", delta=status, delta_color="off")

        with m2:
            wake_sd = variability.get("wake_sd")
            status = "🟢 Good" if wake_sd and wake_sd < 60 else "🔴 High" if wake_sd else ""
            st.metric("Wake SD", f"{wake_sd:.0f} min" if wake_sd else "N/A", delta=status, delta_color="off")

            duration_sd = variability.get("duration_sd")
            status = "🟢 Good" if duration_sd and duration_sd < 60 else "🔴 High" if duration_sd else ""
            st.metric("Duration SD", f"{duration_sd:.0f} min" if duration_sd else "N/A", delta=status, delta_color="off")

        st.caption("🟢 < 60 min = consistent | 🔴 >= 60 min = variable")

    st.divider()

    # Circadian Metrics
    st.header("Circadian Metrics")
    circadian = compute_circadian_metrics(df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sleep Midpoint")
        fig = create_midpoint_chart(circadian.get("midpoints_by_night", []))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Social Jetlag")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Avg Midpoint", circadian.get("avg_midpoint_time", "N/A"))
        with m2:
            st.metric("Weekday Midpoint", circadian.get("weekday_midpoint_time", "N/A"))
        with m3:
            st.metric("Weekend Midpoint", circadian.get("weekend_midpoint_time", "N/A"))

        social_jetlag = circadian.get("social_jetlag_mins")
        if social_jetlag is not None:
            jetlag_hrs = abs(social_jetlag) / 60
            direction = "later" if social_jetlag > 0 else "earlier"
            # Social jetlag > 1 hour is associated with health risks
            status = "🟢 Low" if abs(social_jetlag) < 60 else "🟡 Moderate" if abs(social_jetlag) < 120 else "🔴 High"
            st.metric(
                "Social Jetlag",
                f"{jetlag_hrs:.1f} hrs {direction} on weekends",
                delta=status,
                delta_color="off"
            )
            st.caption("Social jetlag > 1 hr associated with metabolic and cardiovascular risks")
        else:
            st.metric("Social Jetlag", "N/A (need weekday + weekend data)")

    # Two-Process Model
    st.subheader("Two-Process Model (Borbély)")
    st.caption("Sleep regulation emerges from interaction of homeostatic sleep pressure (Process S) and circadian rhythm (Process C). Sleep occurs when S exceeds C.")
    fig = create_two_process_model(df)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Heart Rate and HRV (separate charts)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Heart Rate")
        fig = create_hr_chart(daily_df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("HRV")
        fig = create_hrv_chart(daily_df)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Correlations
    st.header("Correlations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Duration vs REM")
        fig = create_correlation_scatter(
            daily_df, "duration_hours", "rem_minutes",
            "Duration (hours)", "REM (minutes)", COLOR_REM
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Longer sleep = more REM sleep")

    with col2:
        st.subheader("HRV vs Heart Rate")
        fig = create_correlation_scatter(
            daily_df, "hrv_avg", "heart_rate_avg",
            "HRV (ms)", "Heart Rate (bpm)", COLOR_HR
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("HRV vs Awake Time")
        fig = create_correlation_scatter(
            daily_df, "hrv_avg", "awake_minutes",
            "HRV (ms)", "Awake (minutes)", COLOR_AWAKE
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Duration vs Deep Sleep")
        fig = create_correlation_scatter(
            daily_df, "duration_hours", "deep_minutes",
            "Duration (hours)", "Deep (minutes)", COLOR_DEEP
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # AI Analysis
    st.header("AI Analysis")

    sleep_summary = {
        "total_nights": num_nights,
        "avg_duration": f"{avg_dur:.1f}",
        "avg_deep": f"{avg_deep:.0f}",
        "avg_rem": f"{avg_rem:.0f}",
        "avg_hr": f"{avg_hr:.0f}" if not pd.isna(avg_hr) else "N/A",
        "avg_hrv": f"{avg_hrv:.0f}" if not pd.isna(avg_hrv) else "N/A",
    }

    if st.button("Generate Analysis", type="primary"):
        with st.spinner("Analyzing with Qwen..."):
            analysis = get_llm_analysis(sleep_summary)
            st.session_state["analysis"] = analysis

    if "analysis" in st.session_state:
        st.markdown(st.session_state["analysis"])
    else:
        st.info("Click 'Generate Analysis' for AI-powered sleep insights.")



if __name__ == "__main__":
    main()
