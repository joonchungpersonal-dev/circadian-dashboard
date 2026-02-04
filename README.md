# Sleep Dashboard

A real-time sleep analytics dashboard with direct Eight Sleep API integration, providing clinical-grade sleep metrics, circadian rhythm analysis, and correlation analysis.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![OAuth2](https://img.shields.io/badge/Auth-OAuth2-green)

## Preview

![Dashboard Overview](assets/dashboard-1.png)

![Charts and Correlations](assets/dashboard-2.png)

## What Makes This Unique

| Feature | Standard Sleep Apps | This Dashboard |
|---------|---------------------|----------------|
| Data Source | Apple Health export | Direct Eight Sleep OAuth2 API |
| Two-Process Model | Not available | Borbély model visualization (Process S + C) |
| Circadian Metrics | Not available | Sleep midpoint, social jetlag, sunrise markers |
| Sleep Variability | Not available | SD of onset, wake, midpoint, duration |
| Hypnogram | Basic or none | Full stage progression with connecting lines |
| Correlations | Not available | Auto-discovered (Duration↔REM, HRV↔HR) |
| HRV Accuracy | Raw values (often 300+ms) | Filtered 10-150ms for realistic readings |
| Multi-session | Averages all sessions | Aggregates per night, filters main sleep |

## Key Features

### Direct Eight Sleep API Integration
- **Custom OAuth2 client** - Reverse-engineered the new authentication flow after Eight Sleep deprecated session tokens
- **Rate limit handling** - Exponential backoff with automatic retry
- **Token refresh** - Seamless re-authentication on expiry

### Clinical-Grade Sleep Metrics
- **Sleep Schedule Variability** - Standard deviation of sleep timing (used in sleep research)
- **Sleep Midpoint** - Clock time at the middle of sleep period with trend visualization
- **Social Jetlag** - Weekday vs weekend sleep timing difference (associated with health risks)
- **Green/Red indicators** - < 60 min SD = consistent, >= 60 min = variable
- **Main sleep detection** - Filters to 6PM-6AM window, keeps longest session per night

### Circadian Science Visualizations
- **Two-Process Model (Borbély)** - Interactive visualization of sleep regulation showing:
  - Process S (homeostatic sleep pressure) - builds during wake, dissipates during sleep
  - Process C (circadian alerting signal) - 24-hour rhythm from the SCN
  - Sleep gate identification - where S exceeds C, indicating optimal sleep timing
- **Sunrise Markers** - Location-based sunrise times shown on sleep timeline
- **Sleep Midpoint Trend** - Dashed line connecting nightly midpoints

### Advanced Visualizations
- **Hypnogram** - Step chart showing sleep stage progression with connecting lines
- **Correlation scatter plots** - Trend lines with r-values
- **Apple-style dark theme** - Professional UI matching iOS aesthetics

## Quick Start

```bash
# Clone and setup
git clone https://github.com/joonchungpersonal-dev/sleep-dashboard.git
cd sleep-dashboard
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure credentials
echo "EIGHT_SLEEP_EMAIL=your@email.com" > .env
echo "EIGHT_SLEEP_PASSWORD=yourpassword" >> .env
echo "EIGHT_SLEEP_ENABLED=true" >> .env

# Optional: Set location for sunrise calculations (default: Boston)
echo "LATITUDE=42.3601" >> .env
echo "LONGITUDE=-71.0589" >> .env

# Run
streamlit run src/dashboard/app.py
```

## Architecture

```
sleep-dashboard/
├── src/
│   ├── dashboard/
│   │   └── app.py              # Streamlit dashboard
│   └── ingestion/
│       └── eight_sleep_api.py  # Custom OAuth2 API client
├── config/
│   └── settings.py             # Environment configuration
└── docs/
    ├── TECHNICAL.md            # Implementation details
    └── SKILLS.md               # Technologies demonstrated
```

## Dashboard Sections

1. **Last Night** - Duration, deep/REM (with %), HR, HRV
2. **30-Day Averages** - Aggregated metrics across all nights
3. **Sleep Timeline** - 14-day horizontal bar chart with sunrise markers
4. **Hypnogram** - Last night's sleep stage progression
5. **Sleep Variability** - SD metrics with good/high indicators
6. **Circadian Metrics** - Sleep midpoint trend (dashed line), social jetlag
7. **Two-Process Model** - Interactive Borbély model visualization
8. **HR & HRV Trends** - Separate charts with 7-day rolling averages
9. **Correlations** - Duration↔REM, HRV↔HR, HRV↔Awake, Duration↔Deep
10. **AI Analysis** - Optional Qwen-powered insights (requires Ollama)

## Technical Highlights

See [docs/TECHNICAL.md](docs/TECHNICAL.md) for implementation details.
See [docs/SKILLS.md](docs/SKILLS.md) for demonstrated technologies.

## License

MIT License
