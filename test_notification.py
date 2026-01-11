"""
Test script to verify ntfy.sh notifications work
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

NTFY_TOPIC = os.getenv("NTFY_TOPIC")
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"

def send_test_notification():
    """Send a test notification to verify ntfy.sh works."""

    message = """TEST NOTIFICATION

This is a test from your Stock Monitor!

If you see this, notifications are working correctly.

Your phone will receive alerts like this when a Strong Buy signal is detected.
"""

    headers = {
        "Title": "Stock Monitor - Test Alert",
        "Priority": "high",
        "Tags": "white_check_mark,test_tube"
    }

    try:
        print(f"Sending test notification to: {NTFY_URL}")
        response = requests.post(NTFY_URL, data=message.encode(), headers=headers, timeout=10)

        if response.status_code in [200, 201]:
            print("✅ SUCCESS! Test notification sent!")
            print(f"   Check your phone - you should see the notification.")
            return True
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Testing ntfy.sh Notification")
    print("="*60)
    print(f"Topic: {NTFY_TOPIC}")
    print("-"*60)
    send_test_notification()
    print("="*60)
