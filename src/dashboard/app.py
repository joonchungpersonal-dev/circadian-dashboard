"""Circadian Analytics Dashboard — Apple Watch vs Eight Sleep Comparison."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import pytz
import logging

import sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import (
    GCS_BUCKET_NAME,
    COLOR_APPLE_WATCH,
    COLOR_EIGHT_SLEEP,
    COLOR_NEUTRAL,
    COLOR_SUCCESS,
    COLOR_WARNING,
    DEFAULT_DATE_RANGE_DAYS,
    DISCREPANCY_THRESHOLD_MINUTES,
    LOCAL_TIMEZONE,
    EIGHT_SLEEP_EMAIL,
    EIGHT_SLEEP_PASSWORD,
    EIGHT_SLEEP_ENABLED,
)
from src.ingestion.healthkit_loader import HealthKitLoader
from src.analysis.device_comparator import DeviceComparator
from src.ingestion.eight_sleep_api import EightSleepAPI, get_current_sleep_status

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Circadian Analytics",
    page_icon="🌙",
    layout="wide"
)


LOCAL_CACHE_PATH = Path(__file__).parent.parent.parent / "cache" / "healthkit_data.parquet"


@st.cache_data(ttl=300, show_spinner=False)  # 5 min cache for real-time data
def load_eight_sleep_realtime(email: str, password: str) -> dict:
    """Load current Eight Sleep session data via API."""
    if not email or not password:
        return {}
    try:
        return get_current_sleep_status(email, password, LOCAL_TIMEZONE)
    except Exception as e:
        logger.error(f"Eight Sleep API error: {e}")
        return {"error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(start_date: date, end_date: date) -> pd.DataFrame:
    """Load HealthKit data from GCS with Streamlit caching and deduplication."""
    try:
        loader = HealthKitLoader(bucket_name=GCS_BUCKET_NAME)
        # Load 10 recent exports for complete coverage with deduplication
        # Auto Export includes historical data, but spread across exports
        return loader.load_date_range(start_date, end_date, max_exports=10)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def create_sleep_raster_plot(sessions_df: pd.DataFrame, device: str, color: str, title: str) -> go.Figure:
    """
    Create Apple Watch / Eight Sleep style horizontal sleep bars.

    X-axis: Time of day (9 PM to 11 AM, typical sleep window)
    Y-axis: Each date as a row (most recent at top)

    Style inspired by Apple Health and Eight Sleep apps:
    - Clean horizontal bars with rounded appearance
    - Dark background option
    - Subtle grid lines
    - Modern typography
    """
    # Color schemes inspired by Apple Watch (blue/indigo) and Eight Sleep (teal/green)
    if "apple" in device.lower() or "watch" in device.lower():
        bar_color = "#5E5CE6"  # Apple indigo
        bar_gradient = ["#5E5CE6", "#7D7AFF"]  # Subtle gradient feel
        accent_color = "#5E5CE6"
    else:
        bar_color = "#30D158"  # Apple green / Eight Sleep teal
        bar_gradient = ["#30D158", "#32DE84"]
        accent_color = "#30D158"

    fig = go.Figure()

    if sessions_df.empty:
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

    # Filter to device
    device_df = sessions_df[sessions_df["device"] == device].copy()
    if device_df.empty:
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

    # Sort by date (newest first for top position)
    device_df = device_df.sort_values("night_date", ascending=False)

    def to_decimal_hour(ts):
        """Convert timestamp to decimal hour (0-24 scale, adjusted for overnight)."""
        if pd.isna(ts):
            return None
        local_ts = ts.astimezone(local_tz)
        hour = local_ts.hour + local_ts.minute / 60 + local_ts.second / 3600
        # Shift times after midnight to continue past 24 (e.g., 2 AM = 26)
        if hour < 12:
            hour += 24
        return hour

    # Build all bars data
    dates = []
    starts = []
    durations = []
    hover_texts = []

    for _, row in device_df.iterrows():
        night = row["night_date"]
        sleep_start = to_decimal_hour(row.get("sleep_start"))
        sleep_end = to_decimal_hour(row.get("sleep_end"))

        if sleep_start is None or sleep_end is None:
            continue

        duration = row.get("duration_hours", 0)

        # Format for display
        start_ts = row.get("sleep_start")
        end_ts = row.get("sleep_end")
        start_str = start_ts.astimezone(local_tz).strftime("%-I:%M %p") if pd.notna(start_ts) else "N/A"
        end_str = end_ts.astimezone(local_tz).strftime("%-I:%M %p") if pd.notna(end_ts) else "N/A"

        # Date label (e.g., "Mon 2/3")
        date_label = night.strftime("%a %-m/%-d")

        dates.append(date_label)
        starts.append(sleep_start)
        durations.append(sleep_end - sleep_start)
        hover_texts.append(
            f"<b>{night.strftime('%A, %b %-d')}</b><br>"
            f"<span style='color:{accent_color}'>●</span> {start_str} → {end_str}<br>"
            f"<b>{duration:.1f} hours</b>"
        )

    # Single trace for all bars (more efficient)
    fig.add_trace(go.Bar(
        y=dates,
        x=durations,
        base=starts,
        orientation="h",
        marker=dict(
            color=bar_color,
            line=dict(width=0),
            cornerradius=6,  # Rounded corners like Apple
        ),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
        showlegend=False,
    ))

    num_nights = len(dates)

    # Time axis configuration (9 PM to 11 AM range)
    tick_vals = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    tick_text = ["9p", "10p", "11p", "12a", "1a", "2a", "3a", "4a", "5a", "6a", "7a", "8a", "9a", "10a", "11a"]

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color="#FFFFFF", family="SF Pro Display, -apple-system, sans-serif"),
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(
            title="",
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=10, color="#8E8E93"),
            range=[20.5, 35.5],
            gridcolor="rgba(255,255,255,0.06)",
            gridwidth=1,
            zeroline=False,
            showline=False,
            side="top",
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11, color="#FFFFFF", family="SF Pro Text, -apple-system, sans-serif"),
            categoryorder="array",
            categoryarray=dates,  # Keep newest at top
            gridcolor="rgba(255,255,255,0.03)",
            gridwidth=1,
            zeroline=False,
            showline=False,
        ),
        barmode="overlay",
        bargap=0.35,
        height=max(250, num_nights * 32 + 60),
        margin=dict(l=75, r=15, t=45, b=15),
        plot_bgcolor="#1C1C1E",  # Apple dark mode background
        paper_bgcolor="#1C1C1E",
        hoverlabel=dict(
            bgcolor="#2C2C2E",
            bordercolor="#3A3A3C",
            font=dict(size=12, color="#FFFFFF", family="SF Pro Text, -apple-system, sans-serif"),
        ),
    )

    # Add subtle midnight reference line
    fig.add_vline(
        x=24,
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
    )

    return fig


def create_hrv_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Create HRV time series and scatter comparison."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("HRV Over Time", "Device Comparison"))

    if comparison_df.empty or "hrv_mean_watch" not in comparison_df.columns:
        return fig

    df = comparison_df.dropna(subset=["hrv_mean_watch", "hrv_mean_eight"])
    if df.empty:
        return fig

    # Time series
    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hrv_mean_watch"],
        mode="lines+markers",
        name="Apple Watch",
        line=dict(color=COLOR_APPLE_WATCH),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hrv_mean_eight"],
        mode="lines+markers",
        name="Eight Sleep",
        line=dict(color=COLOR_EIGHT_SLEEP),
    ), row=1, col=1)

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df["hrv_mean_watch"],
        y=df["hrv_mean_eight"],
        mode="markers",
        marker=dict(color=df["agreement_score"], colorscale="RdYlGn", showscale=True),
        showlegend=False,
        hovertemplate="Watch: %{x:.1f}ms<br>Eight: %{y:.1f}ms<extra></extra>",
    ), row=1, col=2)

    # Unity line
    max_hrv = max(df["hrv_mean_watch"].max(), df["hrv_mean_eight"].max())
    min_hrv = min(df["hrv_mean_watch"].min(), df["hrv_mean_eight"].min())
    fig.add_trace(go.Scatter(
        x=[min_hrv, max_hrv],
        y=[min_hrv, max_hrv],
        mode="lines",
        line=dict(color=COLOR_NEUTRAL, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="HRV (ms)", row=1, col=1)
    fig.update_xaxes(title_text="Apple Watch HRV (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Eight Sleep HRV (ms)", row=1, col=2)

    fig.update_layout(height=400, showlegend=True)

    return fig


def get_llm_sleep_analysis(sleep_summary: dict) -> str:
    """Get sleep analysis and recommendations from local Qwen model via Ollama."""
    try:
        prompt = f"""You are a sleep health expert. Analyze the following 30-day sleep data and provide:
1. An overall evaluation of sleep quality and consistency
2. Key observations about sleep patterns
3. Specific, actionable recommendations for improvement

Sleep Data Summary:
- Total nights tracked: {sleep_summary.get('total_nights', 'N/A')}
- Average sleep duration: {sleep_summary.get('avg_duration', 'N/A')} hours
- Sleep duration standard deviation: {sleep_summary.get('duration_sd', 'N/A')} minutes
- Average sleep onset time: {sleep_summary.get('avg_onset', 'N/A')}
- Sleep onset standard deviation: {sleep_summary.get('onset_sd', 'N/A')} minutes
- Average wake time: {sleep_summary.get('avg_wake', 'N/A')}
- Wake time standard deviation: {sleep_summary.get('wake_sd', 'N/A')} minutes
- Average midpoint: {sleep_summary.get('avg_midpoint', 'N/A')}
- Midpoint standard deviation: {sleep_summary.get('midpoint_sd', 'N/A')} minutes

Provide your analysis in a clear, organized format with headers. Be specific and practical in your recommendations."""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:14b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": 1000}
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response from model")
        else:
            return f"Error: Model returned status {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "⚠️ Could not connect to Ollama. Make sure Ollama is running with: `ollama serve`"
    except requests.exceptions.Timeout:
        return "⚠️ Model request timed out. The model may be loading or overloaded."
    except Exception as e:
        return f"⚠️ Error getting analysis: {str(e)}"


def create_hr_comparison_chart(comparison_df: pd.DataFrame) -> go.Figure:
    """Create heart rate comparison charts."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Resting HR Over Time", "Device Comparison"))

    if comparison_df.empty or "hr_resting_watch" not in comparison_df.columns:
        return fig

    df = comparison_df.dropna(subset=["hr_resting_watch", "hr_resting_eight"])
    if df.empty:
        return fig

    # Time series
    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hr_resting_watch"],
        mode="lines+markers",
        name="Apple Watch",
        line=dict(color=COLOR_APPLE_WATCH),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hr_resting_eight"],
        mode="lines+markers",
        name="Eight Sleep",
        line=dict(color=COLOR_EIGHT_SLEEP),
    ), row=1, col=1)

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=df["hr_resting_watch"],
        y=df["hr_resting_eight"],
        mode="markers",
        marker=dict(color=df["agreement_score"], colorscale="RdYlGn", showscale=True),
        showlegend=False,
    ), row=1, col=2)

    # Unity line
    max_hr = max(df["hr_resting_watch"].max(), df["hr_resting_eight"].max())
    min_hr = min(df["hr_resting_watch"].min(), df["hr_resting_eight"].min())
    fig.add_trace(go.Scatter(
        x=[min_hr, max_hr],
        y=[min_hr, max_hr],
        mode="lines",
        line=dict(color=COLOR_NEUTRAL, dash="dash"),
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="HR (bpm)", row=1, col=1)
    fig.update_xaxes(title_text="Apple Watch HR (bpm)", row=1, col=2)
    fig.update_yaxes(title_text="Eight Sleep HR (bpm)", row=1, col=2)

    fig.update_layout(height=400)

    return fig


def create_hrv_unified_chart(hrv_df: pd.DataFrame) -> go.Figure:
    """Create HRV time series chart from unified data (no device comparison)."""
    fig = go.Figure()

    if hrv_df.empty:
        fig.add_annotation(
            text="No HRV data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14)
        )
        return fig

    # Sort by date
    df = hrv_df.sort_values("night_date")

    # Mean HRV line
    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hrv_mean"],
        mode="lines+markers",
        name="Mean HRV",
        line=dict(color="#8B5CF6", width=2),
        marker=dict(size=6),
    ))

    # Min/Max range
    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hrv_max"],
        mode="lines",
        name="Max",
        line=dict(color="#8B5CF6", width=0),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["hrv_min"],
        mode="lines",
        name="Min",
        line=dict(color="#8B5CF6", width=0),
        fill="tonexty",
        fillcolor="rgba(139, 92, 246, 0.2)",
        showlegend=False,
    ))

    # Add 7-day rolling average
    if len(df) >= 7:
        df["hrv_rolling"] = df["hrv_mean"].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["night_date"],
            y=df["hrv_rolling"],
            mode="lines",
            name="7-day Avg",
            line=dict(color="#F59E0B", width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Heart Rate Variability (All Sources)",
        xaxis_title="Date",
        yaxis_title="HRV (ms)",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_hr_by_device_chart(hr_df: pd.DataFrame) -> go.Figure:
    """Create heart rate chart showing data by device (works with partial data)."""
    fig = go.Figure()

    if hr_df.empty:
        fig.add_annotation(
            text="No heart rate data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14)
        )
        return fig

    df = hr_df.sort_values("night_date")

    # Check which devices have data
    has_watch = "hr_resting_apple_watch" in df.columns and df["hr_resting_apple_watch"].notna().any()
    has_eight = "hr_resting_eight_sleep" in df.columns and df["hr_resting_eight_sleep"].notna().any()

    if has_watch:
        fig.add_trace(go.Scatter(
            x=df["night_date"],
            y=df["hr_resting_apple_watch"],
            mode="lines+markers",
            name="Apple Watch",
            line=dict(color=COLOR_APPLE_WATCH, width=2),
            marker=dict(size=5),
        ))

    if has_eight:
        fig.add_trace(go.Scatter(
            x=df["night_date"],
            y=df["hr_resting_eight_sleep"],
            mode="lines+markers",
            name="Eight Sleep",
            line=dict(color=COLOR_EIGHT_SLEEP, width=2),
            marker=dict(size=5),
        ))

    # If neither has device data, check for 'other' device
    if not has_watch and not has_eight:
        other_cols = [c for c in df.columns if c.startswith("hr_resting_") and df[c].notna().any()]
        for col in other_cols:
            device_name = col.replace("hr_resting_", "").replace("_", " ").title()
            fig.add_trace(go.Scatter(
                x=df["night_date"],
                y=df[col],
                mode="lines+markers",
                name=device_name,
                line=dict(width=2),
                marker=dict(size=5),
            ))

    fig.update_layout(
        title="Resting Heart Rate by Device",
        xaxis_title="Date",
        yaxis_title="Heart Rate (bpm)",
        height=350,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def calc_time_std_minutes(sessions_df: pd.DataFrame, time_col: str) -> float:
    """Calculate standard deviation of time in minutes."""
    if sessions_df.empty or time_col not in sessions_df.columns:
        return float('nan')
    times = sessions_df[time_col].dropna()
    if len(times) < 2:
        return float('nan')
    local_tz = pytz.timezone(LOCAL_TIMEZONE)
    minutes = []
    for ts in times:
        local_ts = ts.astimezone(local_tz)
        mins = local_ts.hour * 60 + local_ts.minute
        # Handle times after midnight (shift by 24h for consistency)
        if mins < 360:  # Before 6 AM
            mins += 1440
        minutes.append(mins)
    return np.std(minutes)


def calc_duration_std_minutes(sessions_df: pd.DataFrame) -> float:
    """Calculate standard deviation of sleep duration in minutes."""
    if sessions_df.empty or "duration_hours" not in sessions_df.columns:
        return float('nan')
    durations = sessions_df["duration_hours"].dropna() * 60  # Convert to minutes
    if len(durations) < 2:
        return float('nan')
    return np.std(durations)


def main():
    """Main dashboard application."""

    # Sidebar
    with st.sidebar:
        st.title("🌙 Circadian Analytics")
        st.caption("Apple Watch vs Eight Sleep")

        st.divider()

        # Date range selector
        st.subheader("Date Range")

        preset = st.selectbox(
            "Quick Select",
            ["Last 30 days", "Last 90 days", "Last 7 days", "All time", "Custom"],
            index=0  # Default to 30 days
        )

        today = date.today()
        if preset == "Last 7 days":
            start_date = today - timedelta(days=7)
            end_date = today
        elif preset == "Last 30 days":
            start_date = today - timedelta(days=30)
            end_date = today
        elif preset == "Last 90 days":
            start_date = today - timedelta(days=90)
            end_date = today
        elif preset == "All time":
            start_date = date(2020, 1, 1)  # Far past
            end_date = today
        else:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start", today - timedelta(days=30))
            with col2:
                end_date = st.date_input("End", today)

        st.divider()

        # Eight Sleep API Configuration
        st.subheader("Eight Sleep API")

        if EIGHT_SLEEP_EMAIL:
            st.success("✓ Connected")
            st.caption(f"Account: {EIGHT_SLEEP_EMAIL[:3]}...{EIGHT_SLEEP_EMAIL.split('@')[0][-2:]}@...")
        else:
            with st.popover("Configure"):
                st.caption("Enter your Eight Sleep credentials for real-time data")
                email = st.text_input("Email", key="eight_sleep_email_input")
                password = st.text_input("Password", type="password", key="eight_sleep_password_input")
                if st.button("Connect"):
                    st.session_state["eight_sleep_email"] = email
                    st.session_state["eight_sleep_password"] = password
                    st.rerun()
                st.caption("Or set EIGHT_SLEEP_EMAIL and EIGHT_SLEEP_PASSWORD environment variables")

        st.divider()

        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    # Load data
    with st.spinner("Loading data (first load may take a minute, then cached)..."):
        raw_df = load_data(start_date, end_date)

    if raw_df.empty:
        st.warning("No data found for the selected date range.")
        st.info("Make sure Auto Export is configured and has synced data.")
        return

    # Create comparator
    comparator = DeviceComparator(raw_df)
    comparison_df = comparator.build_comparison_df()
    summary = comparator.get_summary_statistics()

    # Display data status in sidebar
    with st.sidebar:
        st.subheader("Data Status")
        st.metric("Total Records", len(raw_df))
        st.metric("Nights with Both Devices", summary.get("nights_both_devices", 0))

        sources = comparator.identify_sources()
        if sources["apple_watch"]:
            st.caption(f"Watch: {sources['apple_watch'][0]}")
        if sources["eight_sleep"]:
            st.caption(f"Eight: {sources['eight_sleep'][0]}")

        st.divider()

        # Download
        st.subheader("Export")
        if not comparison_df.empty:
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "sleep_comparison.csv",
                "text/csv"
            )

    # Get sleep sessions for each device (used across tabs)
    watch_sessions = comparator.extract_sleep_sessions("apple_watch")
    eight_sessions = comparator.extract_sleep_sessions("eight_sleep")

    # Get biometric data
    hrv_unified = comparator.extract_hrv_unified()
    hr_by_device = comparator.extract_heart_rate_by_device()

    # Main content with tabs - Dashboard first
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Sleep Details", "💓 Biometrics", "🤖 Analysis"])

    with tab1:
        # ===== DASHBOARD AT A GLANCE =====

        # Eight Sleep Real-time Status (if enabled)
        eight_sleep_email = st.session_state.get("eight_sleep_email", EIGHT_SLEEP_EMAIL)
        eight_sleep_password = st.session_state.get("eight_sleep_password", EIGHT_SLEEP_PASSWORD)

        if eight_sleep_email and eight_sleep_password:
            with st.expander("🛏️ Eight Sleep Live Status", expanded=False):
                realtime = load_eight_sleep_realtime(eight_sleep_email, eight_sleep_password)

                if "error" in realtime:
                    st.error(f"Eight Sleep API: {realtime['error']}")
                elif realtime:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        stage = realtime.get("sleep_stage", "Unknown")
                        stage_emoji = {"awake": "👁️", "light": "😴", "deep": "🌙", "rem": "💭", "out": "🚶"}.get(str(stage).lower(), "❓")
                        st.metric("Sleep Stage", f"{stage_emoji} {stage}")
                    with col2:
                        hr = realtime.get("heart_rate")
                        st.metric("Heart Rate", f"{hr} bpm" if hr else "N/A")
                    with col3:
                        hrv = realtime.get("hrv")
                        st.metric("HRV", f"{hrv} ms" if hrv else "N/A")
                    with col4:
                        bed_temp = realtime.get("bed_temp")
                        st.metric("Bed Temp", f"{bed_temp}°F" if bed_temp else "N/A")
                    with col5:
                        time_slept = realtime.get("time_slept_minutes", 0)
                        hours = time_slept // 60
                        mins = time_slept % 60
                        st.metric("Time Asleep", f"{hours}h {mins}m" if time_slept else "N/A")

                    # Sleep breakdown if available
                    breakdown = realtime.get("sleep_breakdown", {})
                    if breakdown:
                        st.caption("Current Session Breakdown")
                        cols = st.columns(4)
                        cols[0].metric("Deep", f"{breakdown.get('deep', 0):.0f} min")
                        cols[1].metric("REM", f"{breakdown.get('rem', 0):.0f} min")
                        cols[2].metric("Light", f"{breakdown.get('light', 0):.0f} min")
                        cols[3].metric("Awake", f"{breakdown.get('awake', 0):.0f} min")
                else:
                    st.info("No active sleep session or unable to connect.")

        # Key metrics row
        st.subheader("Last Night")

        # Get most recent night's data
        primary_sessions = eight_sessions if not eight_sessions.empty else watch_sessions
        if not primary_sessions.empty:
            last_night = primary_sessions.iloc[0]  # Most recent (already sorted newest first)
            local_tz = pytz.timezone(LOCAL_TIMEZONE)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                duration = last_night.get("duration_hours", 0)
                st.metric("Sleep Duration", f"{duration:.1f}h")
            with col2:
                start_ts = last_night.get("sleep_start")
                if pd.notna(start_ts):
                    start_str = start_ts.astimezone(local_tz).strftime("%-I:%M %p")
                    st.metric("Bedtime", start_str)
            with col3:
                end_ts = last_night.get("sleep_end")
                if pd.notna(end_ts):
                    end_str = end_ts.astimezone(local_tz).strftime("%-I:%M %p")
                    st.metric("Wake Time", end_str)
            with col4:
                # Latest HRV
                if not hrv_unified.empty:
                    latest_hrv = hrv_unified.iloc[0]["hrv_mean"] if len(hrv_unified) > 0 else None
                    if latest_hrv:
                        st.metric("HRV", f"{latest_hrv:.0f} ms")

        st.divider()

        # 30-day averages
        st.subheader("30-Day Averages")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if not primary_sessions.empty:
                avg_duration = primary_sessions["duration_hours"].mean()
                st.metric("Avg Sleep", f"{avg_duration:.1f}h")
        with col2:
            if not hrv_unified.empty:
                avg_hrv = hrv_unified["hrv_mean"].mean()
                st.metric("Avg HRV", f"{avg_hrv:.0f} ms")
        with col3:
            # Resting HR from Apple Watch or Eight Sleep
            if not hr_by_device.empty:
                hr_col = None
                if "hr_resting_apple_watch" in hr_by_device.columns:
                    hr_col = "hr_resting_apple_watch"
                elif "hr_resting_eight_sleep" in hr_by_device.columns:
                    hr_col = "hr_resting_eight_sleep"
                if hr_col:
                    avg_hr = hr_by_device[hr_col].mean()
                    if not pd.isna(avg_hr):
                        st.metric("Avg Resting HR", f"{avg_hr:.0f} bpm")
        with col4:
            # Sleep consistency (midpoint SD)
            if not primary_sessions.empty:
                midpoint_sd = calc_time_std_minutes(primary_sessions, "midpoint")
                if not np.isnan(midpoint_sd):
                    st.metric("Schedule Consistency", f"±{midpoint_sd:.0f} min")
        with col5:
            st.metric("Nights Tracked", len(primary_sessions) if not primary_sessions.empty else 0)

        st.divider()

        # Sleep timeline - single view showing most data
        st.subheader("Sleep Timeline")

        # Show whichever device has more data, or both if similar
        if len(eight_sessions) > len(watch_sessions) * 2:
            # Eight Sleep has significantly more data
            fig = create_sleep_raster_plot(eight_sessions, "eight_sleep", COLOR_EIGHT_SLEEP, "Eight Sleep")
            st.plotly_chart(fig, use_container_width=True)
        elif len(watch_sessions) > len(eight_sessions) * 2:
            # Apple Watch has significantly more data
            fig = create_sleep_raster_plot(watch_sessions, "apple_watch", COLOR_APPLE_WATCH, "Apple Watch")
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show both side by side
            col1, col2 = st.columns(2)
            with col1:
                fig = create_sleep_raster_plot(watch_sessions, "apple_watch", COLOR_APPLE_WATCH, "Apple Watch")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_sleep_raster_plot(eight_sessions, "eight_sleep", COLOR_EIGHT_SLEEP, "Eight Sleep")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Quick biometrics charts
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("HRV Trend")
            fig = create_hrv_unified_chart(hrv_unified)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Resting Heart Rate")
            fig = create_hr_by_device_chart(hr_by_device)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Sleep Timing Details")

        # Side-by-side raster plots
        col1, col2 = st.columns(2)

        with col1:
            fig_watch = create_sleep_raster_plot(
                watch_sessions, "apple_watch", COLOR_APPLE_WATCH, "Apple Watch"
            )
            st.plotly_chart(fig_watch, use_container_width=True)

        with col2:
            fig_eight = create_sleep_raster_plot(
                eight_sessions, "eight_sleep", COLOR_EIGHT_SLEEP, "Eight Sleep"
            )
            st.plotly_chart(fig_eight, use_container_width=True)

        st.divider()

        # Summary stats
        if not comparison_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                mean_diff = comparison_df["diff_midpoint_min"].mean()
                st.metric(
                    "Avg Midpoint Difference",
                    f"{mean_diff:+.0f} min" if not pd.isna(mean_diff) else "N/A"
                )
            with col2:
                mean_dur_diff = comparison_df["diff_duration_min"].mean()
                st.metric(
                    "Avg Duration Difference",
                    f"{mean_dur_diff:+.0f} min" if not pd.isna(mean_dur_diff) else "N/A"
                )
            with col3:
                st.metric(
                    "Nights Compared",
                    len(comparison_df)
                )
        else:
            st.info("Select a date range with data from both devices to see comparison metrics.")

    with tab3:
        st.header("Biometrics & Circadian Regularity")

        # Circadian regularity metrics (SD of sleep timing)
        st.subheader("🕐 Sleep Timing Variability")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Apple Watch**")
            watch_onset_std = calc_time_std_minutes(watch_sessions, "sleep_start")
            watch_midpoint_std = calc_time_std_minutes(watch_sessions, "midpoint")
            watch_wake_std = calc_time_std_minutes(watch_sessions, "sleep_end")
            watch_duration_std = calc_duration_std_minutes(watch_sessions)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Onset SD", f"{watch_onset_std:.0f} min" if not np.isnan(watch_onset_std) else "N/A")
            with m2:
                st.metric("Midpoint SD", f"{watch_midpoint_std:.0f} min" if not np.isnan(watch_midpoint_std) else "N/A")
            with m3:
                st.metric("Wake SD", f"{watch_wake_std:.0f} min" if not np.isnan(watch_wake_std) else "N/A")
            with m4:
                st.metric("Duration SD", f"{watch_duration_std:.0f} min" if not np.isnan(watch_duration_std) else "N/A")

        with col2:
            st.markdown("**Eight Sleep**")
            eight_onset_std = calc_time_std_minutes(eight_sessions, "sleep_start")
            eight_midpoint_std = calc_time_std_minutes(eight_sessions, "midpoint")
            eight_wake_std = calc_time_std_minutes(eight_sessions, "sleep_end")
            eight_duration_std = calc_duration_std_minutes(eight_sessions)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Onset SD", f"{eight_onset_std:.0f} min" if not np.isnan(eight_onset_std) else "N/A")
            with m2:
                st.metric("Midpoint SD", f"{eight_midpoint_std:.0f} min" if not np.isnan(eight_midpoint_std) else "N/A")
            with m3:
                st.metric("Wake SD", f"{eight_wake_std:.0f} min" if not np.isnan(eight_wake_std) else "N/A")
            with m4:
                st.metric("Duration SD", f"{eight_duration_std:.0f} min" if not np.isnan(eight_duration_std) else "N/A")

        st.divider()

        # HRV and HR charts - detailed view
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Heart Rate Variability")
            fig = create_hrv_unified_chart(hrv_unified)
            st.plotly_chart(fig, use_container_width=True)
            if not hrv_unified.empty:
                avg_hrv = hrv_unified["hrv_mean"].mean()
                min_hrv = hrv_unified["hrv_min"].min()
                max_hrv = hrv_unified["hrv_max"].max()
                c1, c2, c3 = st.columns(3)
                c1.metric("Average", f"{avg_hrv:.0f} ms")
                c2.metric("Min", f"{min_hrv:.0f} ms")
                c3.metric("Max", f"{max_hrv:.0f} ms")

        with col2:
            st.subheader("Resting Heart Rate")
            fig = create_hr_by_device_chart(hr_by_device)
            st.plotly_chart(fig, use_container_width=True)
            if not hr_by_device.empty:
                hr_cols = [c for c in hr_by_device.columns if c.startswith("hr_resting_") and "other" not in c]
                cols = st.columns(len(hr_cols)) if hr_cols else []
                for i, col in enumerate(hr_cols):
                    device = col.replace("hr_resting_", "").replace("_", " ").title()
                    avg = hr_by_device[col].mean()
                    if not pd.isna(avg):
                        cols[i].metric(f"{device}", f"{avg:.0f} bpm")

    with tab3:
        st.header("Analysis & Recommendations")

        # LLM Analysis Section
        st.subheader("🤖 AI Sleep Analysis (Qwen)")

        # Prepare sleep summary for LLM
        def format_time(sessions_df, col):
            if sessions_df.empty or col not in sessions_df.columns:
                return "N/A"
            times = sessions_df[col].dropna()
            if len(times) == 0:
                return "N/A"
            local_tz = pytz.timezone(LOCAL_TIMEZONE)
            avg_mins = np.mean([
                (t.astimezone(local_tz).hour * 60 + t.astimezone(local_tz).minute)
                if t.astimezone(local_tz).hour >= 12 else
                (t.astimezone(local_tz).hour * 60 + t.astimezone(local_tz).minute + 1440)
                for t in times
            ])
            hrs = int(avg_mins // 60) % 24
            mins = int(avg_mins % 60)
            return f"{hrs:02d}:{mins:02d}"

        # Use Eight Sleep data preferentially, fall back to Apple Watch
        primary_sessions = eight_sessions if not eight_sessions.empty else watch_sessions

        sleep_summary = {
            "total_nights": len(primary_sessions) if not primary_sessions.empty else 0,
            "avg_duration": f"{primary_sessions['duration_hours'].mean():.1f}" if not primary_sessions.empty else "N/A",
            "duration_sd": f"{primary_sessions['duration_hours'].std() * 60:.0f}" if not primary_sessions.empty else "N/A",
            "avg_onset": format_time(primary_sessions, "sleep_start"),
            "onset_sd": f"{calc_time_std_minutes(primary_sessions, 'sleep_start'):.0f}" if not primary_sessions.empty else "N/A",
            "avg_wake": format_time(primary_sessions, "sleep_end"),
            "wake_sd": f"{calc_time_std_minutes(primary_sessions, 'sleep_end'):.0f}" if not primary_sessions.empty else "N/A",
            "avg_midpoint": format_time(primary_sessions, "midpoint"),
            "midpoint_sd": f"{calc_time_std_minutes(primary_sessions, 'midpoint'):.0f}" if not primary_sessions.empty else "N/A",
        }

        if st.button("🔄 Generate Analysis", type="primary"):
            with st.spinner("Analyzing sleep patterns with Qwen..."):
                analysis = get_llm_sleep_analysis(sleep_summary)
                st.session_state["llm_analysis"] = analysis

        if "llm_analysis" in st.session_state:
            st.markdown(st.session_state["llm_analysis"])
        else:
            st.info("Click 'Generate Analysis' to get AI-powered sleep recommendations.")

        st.divider()

        # Device Agreement Section
        st.subheader("📊 Device Agreement Analysis")

        if comparison_df.empty:
            st.warning("Not enough data for agreement analysis.")
        else:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                mean_agreement = comparison_df["agreement_score"].mean()
                st.metric("Mean Agreement Score", f"{mean_agreement:.0f}%")

            with col2:
                high_agreement = (comparison_df["agreement_score"] > 90).sum()
                st.metric("Nights >90% Agreement", high_agreement)

            with col3:
                mean_dur = abs(comparison_df["diff_duration_min"]).mean()
                st.metric("Mean Duration Diff", f"{mean_dur:.0f} min")

            with col4:
                if "duration_correlation" in summary:
                    st.metric("Duration Correlation", f"{summary['duration_correlation']:.2f}")

            st.divider()

            # Histograms
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    comparison_df,
                    x="diff_duration_min",
                    nbins=20,
                    title="Duration Difference Distribution",
                    labels={"diff_duration_min": "Difference (minutes)"}
                )
                fig.add_vline(x=0, line_dash="dash", line_color=COLOR_NEUTRAL)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    comparison_df,
                    x="diff_midpoint_min",
                    nbins=20,
                    title="Midpoint Difference Distribution",
                    labels={"diff_midpoint_min": "Difference (minutes)"}
                )
                fig.add_vline(x=0, line_dash="dash", line_color=COLOR_NEUTRAL)
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Flagged nights
            st.subheader("⚠️ Nights with Large Discrepancies")

            flagged = comparison_df[
                (abs(comparison_df["diff_duration_min"]) > DISCREPANCY_THRESHOLD_MINUTES) |
                (abs(comparison_df["diff_midpoint_min"]) > DISCREPANCY_THRESHOLD_MINUTES)
            ]

            if flagged.empty:
                st.success("No significant discrepancies found!")
            else:
                display_cols = ["night_date", "duration_hours_watch", "duration_hours_eight",
                              "diff_duration_min", "diff_midpoint_min", "agreement_score"]
                display_cols = [c for c in display_cols if c in flagged.columns]
                st.dataframe(flagged[display_cols], use_container_width=True)

            st.divider()

            # Agreement trend
            st.subheader("Agreement Trend")
            fig = px.line(
                comparison_df,
                x="night_date",
                y="agreement_score",
                title="Agreement Score Over Time"
            )
            # Add 7-day rolling average
            if len(comparison_df) > 7:
                comparison_df["rolling_agreement"] = comparison_df["agreement_score"].rolling(7).mean()
                fig.add_trace(go.Scatter(
                    x=comparison_df["night_date"],
                    y=comparison_df["rolling_agreement"],
                    mode="lines",
                    name="7-day Average",
                    line=dict(color=COLOR_NEUTRAL, dash="dash")
                ))
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
