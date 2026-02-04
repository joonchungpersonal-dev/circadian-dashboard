#!/usr/bin/env python3
"""Test Eight Sleep API connection.

Run with:
    python test_eight_sleep.py your-email@example.com your-password
"""

import asyncio
import sys
from pyeight.eight import EightSleep


async def test_connection(email: str, password: str):
    """Test Eight Sleep API connection and data retrieval."""
    print(f"Testing Eight Sleep API connection...")
    print(f"Email: {email[:3]}...{email.split('@')[0][-2:]}@{email.split('@')[1]}")
    print()

    eight = EightSleep(email, password, "America/New_York")

    try:
        print("1. Starting connection...")
        await eight.start()
        print("   ✓ Connected successfully")

        print("2. Fetching device data...")
        await eight.update_device_data()
        print(f"   ✓ Device ID: {eight.device_id}")
        print(f"   ✓ Is Pod: {eight.is_pod}")

        print("3. Fetching user data...")
        await eight.update_user_data()
        print(f"   ✓ Users found: {len(eight.users)}")

        if eight.users:
            user = next(iter(eight.users.values()))
            print()
            print("=== User Data ===")
            print(f"Current sleep stage: {user.current_sleep_stage}")
            print(f"Current heart rate: {user.current_heart_rate}")
            print(f"Current HRV: {user.current_hrv}")
            print(f"Current bed temp: {user.current_bed_temp}")
            print(f"Current room temp: {user.current_room_temp}")
            print(f"Time slept (min): {user.time_slept}")

            print()
            print("=== Last Session ===")
            print(f"Session date: {user.last_session_date}")
            print(f"Heart rate avg: {user.last_heart_rate}")
            print(f"Sleep breakdown: {user.last_sleep_breakdown}")
            print(f"Fitness score: {user.last_sleep_fitness_score}")

        print()
        print("✅ Eight Sleep API is working!")
        return True

    except Exception as e:
        print()
        print(f"❌ Error: {type(e).__name__}: {e}")

        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("   → Authentication failed. Check your email/password.")
        elif "403" in str(e):
            print("   → Access forbidden. Eight Sleep may have changed their API.")
        elif "timeout" in str(e).lower():
            print("   → Connection timed out. Check your internet connection.")
        else:
            print("   → Unknown error. The API may have changed.")

        return False

    finally:
        await eight.stop()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_eight_sleep.py <email> <password>")
        print()
        print("Example:")
        print("  python test_eight_sleep.py user@example.com mypassword")
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]

    success = asyncio.run(test_connection(email, password))
    sys.exit(0 if success else 1)
