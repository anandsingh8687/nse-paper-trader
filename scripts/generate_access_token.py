"""
=============================================================================
Zerodha Kite Access Token Generator
=============================================================================
Kite access tokens expire daily. This helper script generates a fresh token
via the Kite login flow.

Usage:
    python scripts/generate_access_token.py

This will:
1. Open your browser to the Kite login page
2. After login, capture the request_token from the redirect URL
3. Generate an access_token and print it
4. Optionally update your .env file

NOTE: In production, OpenAlgo handles this via its /zerodha/callback endpoint.
This script is for initial setup and debugging.
=============================================================================
"""

import os
import hashlib
from dotenv import load_dotenv, set_key
from kiteconnect import KiteConnect
from loguru import logger

load_dotenv()


def main():
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")

    if not api_key or not api_secret:
        print("ERROR: Set KITE_API_KEY and KITE_API_SECRET in .env first")
        return

    kite = KiteConnect(api_key=api_key)

    # Step 1: Get login URL
    login_url = kite.login_url()
    print(f"\n1. Open this URL in your browser:\n   {login_url}\n")
    print("2. Log in with your Zerodha credentials")
    print("3. After redirect, copy the 'request_token' from the URL")
    print("   (URL will look like: http://127.0.0.1:5000?request_token=XXXXX&...)\n")

    request_token = input("Paste your request_token here: ").strip()

    if not request_token:
        print("No token provided. Exiting.")
        return

    # Step 2: Generate access token
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data["access_token"]

        print(f"\nAccess Token: {access_token}")
        print(f"Valid until: end of today (tokens expire daily)")

        # Step 3: Update .env
        update = input("\nUpdate .env with this token? (y/n): ").strip().lower()
        if update == "y":
            env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
            if os.path.exists(env_path):
                set_key(env_path, "KITE_ACCESS_TOKEN", access_token)
                print(f"Updated {env_path}")
            else:
                print(f".env not found at {env_path}. Add manually:")
                print(f"KITE_ACCESS_TOKEN={access_token}")

    except Exception as e:
        print(f"\nERROR generating access token: {e}")
        print("Common issues:")
        print("  - request_token already used (they're single-use)")
        print("  - API key/secret mismatch")
        print("  - Kite Connect subscription expired")


if __name__ == "__main__":
    main()
