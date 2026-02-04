"""Eight Sleep Dashboard — Sleep Analytics & Trends."""

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
)
from src.ingestion.eight_sleep_api import (
    EightSleepClient,
    get_current_sleep_status,
    get_eight_sleep_data_sync,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Eight Sleep Dashboard",
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

        date_label = night.strftime("%a %-m/%-d")
        dates.append(date_label)
        starts.append(sleep_start)
        durations.append(sleep_end - sleep_start)

        hover_texts.append(
            f"<b>{night.strftime('%A, %b %-d')}</b><br>"
            f"<b>{duration:.1f} hours</b><br>"
            f"Deep: {deep:.0f}m | REM: {rem:.0f}m | Light: {light:.0f}m"
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


def create_sleep_stages_chart(df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart for sleep stages."""
    fig = go.Figure()

    if df.empty:
        return fig

    df = df.sort_values("night_date", ascending=True).tail(14)

    dates = [d.strftime("%-m/%-d") for d in df["night_date"]]

    fig.add_trace(go.Bar(
        name="Deep",
        x=dates,
        y=df["deep_minutes"] / 60,
        marker_color=COLOR_DEEP,
    ))
    fig.add_trace(go.Bar(
        name="REM",
        x=dates,
        y=df["rem_minutes"] / 60,
        marker_color=COLOR_REM,
    ))
    fig.add_trace(go.Bar(
        name="Light",
        x=dates,
        y=df["light_minutes"] / 60,
        marker_color=COLOR_LIGHT,
    ))
    fig.add_trace(go.Bar(
        name="Awake",
        x=dates,
        y=df["awake_minutes"] / 60,
        marker_color=COLOR_AWAKE,
    ))

    fig.update_layout(
        barmode="stack",
        xaxis_title="",
        yaxis_title="Hours",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=300,
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
        xaxis=dict(tickfont=dict(color="#8E8E93")),
        yaxis=dict(tickfont=dict(color="#8E8E93"), gridcolor="rgba(255,255,255,0.06)"),
    )

    return fig


def create_biometrics_chart(df: pd.DataFrame) -> go.Figure:
    """Create dual-axis chart for HR and HRV."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if df.empty:
        return fig

    df = df.sort_values("night_date", ascending=True)
    df = df.dropna(subset=["heart_rate_avg"])

    if df.empty:
        return fig

    # Heart Rate
    fig.add_trace(
        go.Scatter(
            x=df["night_date"],
            y=df["heart_rate_avg"],
            name="Heart Rate",
            line=dict(color=COLOR_HR, width=2),
            mode="lines+markers",
            marker=dict(size=5),
        ),
        secondary_y=False,
    )

    # HRV
    hrv_df = df.dropna(subset=["hrv_avg"])
    if not hrv_df.empty:
        fig.add_trace(
            go.Scatter(
                x=hrv_df["night_date"],
                y=hrv_df["hrv_avg"],
                name="HRV",
                line=dict(color=COLOR_HRV, width=2),
                mode="lines+markers",
                marker=dict(size=5),
            ),
            secondary_y=True,
        )

    fig.update_xaxes(title_text="", tickfont=dict(color="#8E8E93"))
    fig.update_yaxes(title_text="Heart Rate (bpm)", secondary_y=False, tickfont=dict(color="#8E8E93"))
    fig.update_yaxes(title_text="HRV (ms)", secondary_y=True, tickfont=dict(color="#8E8E93"))

    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=300,
        margin=dict(l=50, r=50, t=40, b=30),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
    )

    return fig


def create_duration_trend(df: pd.DataFrame) -> go.Figure:
    """Create sleep duration trend with 7-day average."""
    fig = go.Figure()

    if df.empty:
        return fig

    df = df.sort_values("night_date", ascending=True)

    fig.add_trace(go.Scatter(
        x=df["night_date"],
        y=df["duration_hours"],
        name="Duration",
        line=dict(color=COLOR_PRIMARY, width=2),
        mode="lines+markers",
        marker=dict(size=5),
    ))

    if len(df) >= 7:
        df["rolling_avg"] = df["duration_hours"].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["night_date"],
            y=df["rolling_avg"],
            name="7-day Avg",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            mode="lines",
        ))

    # 8-hour target line
    fig.add_hline(y=8, line=dict(color="rgba(255,255,255,0.3)", dash="dot"))

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Hours",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=300,
        margin=dict(l=50, r=20, t=40, b=30),
        plot_bgcolor="#1C1C1E",
        paper_bgcolor="#1C1C1E",
        font=dict(color="#FFFFFF"),
        xaxis=dict(tickfont=dict(color="#8E8E93")),
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
        st.title("Eight Sleep")
        st.caption("Sleep Analytics Dashboard")
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

    # === DASHBOARD ===

    # Last Night Summary
    st.header("Last Night")

    last_night = df.iloc[0]
    local_tz = pytz.timezone(LOCAL_TIMEZONE)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        duration = last_night.get("duration_hours", 0)
        st.metric("Sleep Duration", f"{duration:.1f}h")

    with col2:
        deep = last_night.get("deep_minutes", 0) or 0
        st.metric("Deep Sleep", f"{deep:.0f}m")

    with col3:
        rem = last_night.get("rem_minutes", 0) or 0
        st.metric("REM Sleep", f"{rem:.0f}m")

    with col4:
        hr = last_night.get("heart_rate_avg")
        st.metric("Avg Heart Rate", f"{hr:.0f} bpm" if hr else "N/A")

    with col5:
        hrv = last_night.get("hrv_avg")
        st.metric("HRV", f"{hrv:.0f} ms" if hrv else "N/A")

    st.divider()

    # Averages
    st.header(f"{days}-Day Averages")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        avg_dur = df["duration_hours"].mean()
        st.metric("Avg Sleep", f"{avg_dur:.1f}h")

    with col2:
        avg_deep = df["deep_minutes"].mean()
        st.metric("Avg Deep", f"{avg_deep:.0f}m")

    with col3:
        avg_rem = df["rem_minutes"].mean()
        st.metric("Avg REM", f"{avg_rem:.0f}m")

    with col4:
        avg_hr = df["heart_rate_avg"].dropna().mean()
        st.metric("Avg HR", f"{avg_hr:.0f} bpm" if not pd.isna(avg_hr) else "N/A")

    with col5:
        avg_hrv = df["hrv_avg"].dropna().mean()
        st.metric("Avg HRV", f"{avg_hrv:.0f} ms" if not pd.isna(avg_hrv) else "N/A")

    with col6:
        st.metric("Nights Tracked", len(df))

    st.divider()

    # Sleep Timeline
    st.header("Sleep Timeline")
    fig = create_sleep_timeline(df)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sleep Stages")
        fig = create_sleep_stages_chart(df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Duration Trend")
        fig = create_duration_trend(df)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Biometrics
    st.header("Heart Rate & HRV")
    fig = create_biometrics_chart(df)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # AI Analysis
    st.header("AI Analysis")

    sleep_summary = {
        "total_nights": len(df),
        "avg_duration": f"{df['duration_hours'].mean():.1f}",
        "avg_deep": f"{df['deep_minutes'].mean():.0f}",
        "avg_rem": f"{df['rem_minutes'].mean():.0f}",
        "avg_hr": f"{df['heart_rate_avg'].dropna().mean():.0f}" if not df['heart_rate_avg'].dropna().empty else "N/A",
        "avg_hrv": f"{df['hrv_avg'].dropna().mean():.0f}" if not df['hrv_avg'].dropna().empty else "N/A",
    }

    if st.button("Generate Analysis", type="primary"):
        with st.spinner("Analyzing with Qwen..."):
            analysis = get_llm_analysis(sleep_summary)
            st.session_state["analysis"] = analysis

    if "analysis" in st.session_state:
        st.markdown(st.session_state["analysis"])
    else:
        st.info("Click 'Generate Analysis' for AI-powered sleep insights.")

    st.divider()

    # Raw Data
    with st.expander("View Raw Data"):
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
