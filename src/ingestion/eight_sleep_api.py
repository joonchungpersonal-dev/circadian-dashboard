"""Eight Sleep API client using OAuth2 authentication.

Uses the Eight Sleep OAuth2 API for direct data access.
This provides more detailed sleep data than Apple Health export.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional
import aiohttp
import pandas as pd
import pytz

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.settings import LOCAL_TIMEZONE

logger = logging.getLogger(__name__)

# OAuth2 API endpoints
AUTH_URL = "https://auth-api.8slp.net/v1/tokens"
CLIENT_API_URL = "https://client-api.8slp.net/v1"

# OAuth2 client credentials (from lukas-clarke/pyEight)
KNOWN_CLIENT_ID = "0894c7f33bb94800a03f1f4df13a4f38"
KNOWN_CLIENT_SECRET = "f0954a3ed5763ba3d06834c73731a32f15f168f47d4f164751275def86db0c76"

DEFAULT_HEADERS = {
    "content-type": "application/json",
    "user-agent": "okhttp/4.9.3",
    "accept": "application/json",
}


class EightSleepClient:
    """Eight Sleep API client using OAuth2 authentication."""

    def __init__(
        self,
        email: str,
        password: str,
        timezone: str = LOCAL_TIMEZONE,
    ):
        self.email = email
        self.password = password
        self.timezone = timezone
        self.local_tz = pytz.timezone(timezone)
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
        self._user_id: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self):
        """Ensure we have an active session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def login(self, retries: int = 3) -> bool:
        """Authenticate with Eight Sleep using OAuth2."""
        await self._ensure_session()

        auth_data = {
            "client_id": KNOWN_CLIENT_ID,
            "client_secret": KNOWN_CLIENT_SECRET,
            "grant_type": "password",
            "username": self.email,
            "password": self.password,
        }

        for attempt in range(retries):
            try:
                async with self._session.post(AUTH_URL, data=auth_data) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._access_token = data.get("access_token")
                        self._user_id = data.get("userId")
                        expires_in = data.get("expires_in", 72000)
                        self._token_expires = datetime.now() + timedelta(seconds=expires_in - 120)
                        logger.info(f"Eight Sleep OAuth2 login successful, user_id: {self._user_id}")
                        return True
                    elif resp.status == 429:
                        wait_time = 2 ** attempt * 5
                        logger.warning(f"Login rate limited, waiting {wait_time}s")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        text = await resp.text()
                        logger.error(f"OAuth2 login failed: {resp.status} - {text}")
                        return False
            except Exception as e:
                logger.error(f"Login error: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False
        return False

    def _token_expired(self) -> bool:
        """Check if the access token has expired."""
        if not self._token_expires:
            return True
        return datetime.now() >= self._token_expires

    async def _api_request(self, endpoint: str, method: str = "GET", data: dict = None, retries: int = 3) -> Optional[dict]:
        """Make authenticated API request with rate limit handling."""
        if not self._access_token or self._token_expired():
            if not await self.login():
                return None

        await self._ensure_session()

        headers = DEFAULT_HEADERS.copy()
        headers["Authorization"] = f"Bearer {self._access_token}"

        url = f"{CLIENT_API_URL}{endpoint}"

        for attempt in range(retries):
            try:
                if method == "GET":
                    async with self._session.get(url, headers=headers) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 401:
                            # Token expired, try re-login
                            self._access_token = None
                            if await self.login():
                                headers["Authorization"] = f"Bearer {self._access_token}"
                                continue
                        elif resp.status == 429:
                            wait_time = 2 ** attempt * 5
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            text = await resp.text()
                            logger.error(f"API error {resp.status}: {text[:200]}")
                elif method == "POST":
                    async with self._session.post(url, headers=headers, json=data) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        elif resp.status == 429:
                            wait_time = 2 ** attempt * 5
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                            await asyncio.sleep(wait_time)
                            continue
            except Exception as e:
                logger.error(f"API request error: {e}")

        return None

    async def get_user_profile(self) -> Optional[dict]:
        """Get user profile and device info."""
        return await self._api_request(f"/users/{self._user_id}")

    async def get_device_info(self) -> Optional[dict]:
        """Get device/pod information."""
        profile = await self.get_user_profile()
        if profile and "user" in profile:
            device_id = profile["user"].get("currentDevice", {}).get("id")
            if device_id:
                return await self._api_request(f"/devices/{device_id}")
        return None

    async def get_intervals(self, start_date: date, end_date: date) -> Optional[dict]:
        """
        Get sleep intervals (sessions) for date range.

        This is the main endpoint for historical sleep data.
        """
        start_str = start_date.strftime("%Y-%m-%dT00:00:00.000Z")
        end_str = end_date.strftime("%Y-%m-%dT23:59:59.999Z")

        return await self._api_request(
            f"/users/{self._user_id}/intervals"
            f"?tz={self.timezone}&from={start_str}&to={end_str}"
        )

    async def get_trends(self, start_date: date, end_date: date) -> Optional[dict]:
        """Get sleep trends/scores for date range."""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        return await self._api_request(
            f"/users/{self._user_id}/trends"
            f"?tz={self.timezone}&from={start_str}&to={end_str}"
        )

    async def get_current_device_status(self) -> Optional[dict]:
        """Get current device status (temperature, etc)."""
        profile = await self.get_user_profile()
        if profile and "user" in profile:
            device_id = profile["user"].get("currentDevice", {}).get("id")
            if device_id:
                return await self._api_request(f"/devices/{device_id}/status")
        return None

    async def close(self):
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_sleep_sessions_df(self, days: int = 30) -> pd.DataFrame:
        """
        Get sleep sessions as DataFrame.

        Args:
            days: Number of days of history

        Returns:
            DataFrame with sleep session data
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        intervals = await self.get_intervals(start_date, end_date)

        if not intervals or "intervals" not in intervals:
            return pd.DataFrame()

        sessions = []
        for interval in intervals.get("intervals", []):
            # Parse timestamps
            ts_str = interval.get("ts")
            if not ts_str:
                continue

            try:
                session_start = pd.to_datetime(ts_str)
                if session_start.tzinfo is None:
                    session_start = self.local_tz.localize(session_start)
                else:
                    # Convert to local timezone for proper night assignment
                    session_start = session_start.astimezone(self.local_tz)
            except Exception:
                continue

            # Get sleep stages breakdown
            stages = interval.get("stages", [])
            stage_durations = {"awake": 0, "light": 0, "deep": 0, "rem": 0}

            sleep_start = None
            sleep_end = None

            for stage in stages:
                stage_type = stage.get("stage", "").lower()
                duration = stage.get("duration", 0)  # seconds

                if stage_type in stage_durations:
                    stage_durations[stage_type] += duration / 60  # to minutes

                # Track sleep start/end from stage timestamps
                stage_ts = stage.get("ts")
                if stage_ts:
                    stage_time = pd.to_datetime(stage_ts)
                    if sleep_start is None or stage_time < sleep_start:
                        sleep_start = stage_time
                    if sleep_end is None or stage_time > sleep_end:
                        sleep_end = stage_time

            # Get timeseries data averages
            timeseries = interval.get("timeseries", {})

            def avg_timeseries(key, min_val=None, max_val=None):
                data = timeseries.get(key, [])
                values = [v[1] for v in data if v[1] is not None]
                # Filter to reasonable range if specified
                if min_val is not None or max_val is not None:
                    values = [v for v in values
                              if (min_val is None or v >= min_val)
                              and (max_val is None or v <= max_val)]
                return sum(values) / len(values) if values else None

            # Get scores
            score = interval.get("score", 0)

            total_sleep_mins = sum(v for k, v in stage_durations.items() if k != "awake")

            # Build stage sequence for hypnogram
            stage_sequence = []
            cumulative_mins = 0
            for stage in stages:
                stage_type = stage.get("stage", "").lower()
                duration_secs = stage.get("duration", 0)
                duration_mins = duration_secs / 60
                stage_sequence.append({
                    "stage": stage_type,
                    "start_min": cumulative_mins,
                    "end_min": cumulative_mins + duration_mins,
                    "duration_min": duration_mins,
                })
                cumulative_mins += duration_mins

            # Determine night_date based on local time:
            # - If session starts before 6 PM, it belongs to previous night (woke up that morning)
            # - If session starts at/after 6 PM, it belongs to that night (going to bed)
            local_hour = session_start.hour
            if local_hour < 18:  # Before 6 PM - assign to previous night
                night_date = session_start.date() - timedelta(days=1)
            else:  # 6 PM or later - assign to current night
                night_date = session_start.date()

            sessions.append({
                "night_date": night_date,
                "session_start": session_start,
                "sleep_start": sleep_start,
                "sleep_end": sleep_end,
                "duration_hours": total_sleep_mins / 60,
                "deep_minutes": stage_durations["deep"],
                "rem_minutes": stage_durations["rem"],
                "light_minutes": stage_durations["light"],
                "awake_minutes": stage_durations["awake"],
                "heart_rate_avg": avg_timeseries("heartRate"),
                "hrv_avg": avg_timeseries("hrv", min_val=10, max_val=150),  # Filter outliers
                "resp_rate_avg": avg_timeseries("respiratoryRate"),
                "bed_temp_avg": avg_timeseries("bedTemperature"),
                "room_temp_avg": avg_timeseries("roomTemperature"),
                "toss_turns": interval.get("tnt", 0),
                "sleep_score": score,
                "source": "eight_sleep_api",
                "stages": stage_sequence,  # Raw stage sequence for hypnogram
            })

        df = pd.DataFrame(sessions)
        if not df.empty:
            df = df.sort_values("night_date", ascending=False)

        return df


async def test_connection(email: str, password: str) -> dict:
    """Test Eight Sleep API connection."""
    client = EightSleepClient(email, password)

    result = {
        "connected": False,
        "user_id": None,
        "device": None,
        "error": None,
    }

    try:
        if await client.login():
            result["connected"] = True
            result["user_id"] = client._user_id

            # Try to get device info
            profile = await client.get_user_profile()
            if profile and "user" in profile:
                device = profile["user"].get("currentDevice", {})
                result["device"] = device.get("type", "Unknown")

            # Try to get recent data
            df = await client.get_sleep_sessions_df(days=7)
            result["sessions_found"] = len(df)

        else:
            result["error"] = "Login failed"
    except Exception as e:
        result["error"] = str(e)
    finally:
        await client.close()

    return result


def get_eight_sleep_data_sync(email: str, password: str, days: int = 30) -> pd.DataFrame:
    """Synchronous wrapper for getting Eight Sleep data."""
    async def _fetch():
        client = EightSleepClient(email, password)
        try:
            await client.login()
            return await client.get_sleep_sessions_df(days=days)
        finally:
            await client.close()

    return asyncio.run(_fetch())


def test_connection_sync(email: str, password: str) -> dict:
    """Synchronous wrapper for testing connection."""
    return asyncio.run(test_connection(email, password))


def get_current_sleep_status(email: str, password: str, timezone: str = LOCAL_TIMEZONE) -> dict:
    """Get current sleep status and device info for real-time dashboard display."""
    async def _fetch():
        client = EightSleepClient(email, password, timezone)
        try:
            if not await client.login():
                return {"error": "Login failed"}

            result = {
                "connected": True,
                "user_id": client._user_id,
            }

            # Get device status
            status = await client.get_current_device_status()
            if status and "result" in status:
                device = status["result"]
                result["device_status"] = {
                    "left_heating_level": device.get("leftHeatingLevel"),
                    "right_heating_level": device.get("rightHeatingLevel"),
                    "left_target_level": device.get("leftTargetHeatingLevel"),
                    "right_target_level": device.get("rightTargetHeatingLevel"),
                    "is_on": device.get("on", False),
                }

            # Get latest sleep session
            df = await client.get_sleep_sessions_df(days=1)
            if not df.empty:
                latest = df.iloc[0]
                result["latest_session"] = {
                    "night_date": str(latest["night_date"]),
                    "duration_hours": latest["duration_hours"],
                    "deep_minutes": latest["deep_minutes"],
                    "rem_minutes": latest["rem_minutes"],
                    "heart_rate_avg": latest["heart_rate_avg"],
                    "hrv_avg": latest["hrv_avg"],
                    "sleep_score": latest["sleep_score"],
                }

            return result
        except Exception as e:
            return {"error": str(e)}
        finally:
            await client.close()

    return asyncio.run(_fetch())


# Alias for backwards compatibility
EightSleepAPI = EightSleepClient
