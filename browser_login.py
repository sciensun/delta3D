import argparse
import os

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description="Open a persistent Playwright browser profile for manual website login.")
    parser.add_argument("--url", default="https://sketchfab.com/login")
    parser.add_argument("--browser_profile", default=".browser_profile")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if not args.headless and not os.environ.get("DISPLAY"):
        raise SystemExit(
            "No DISPLAY is available. Start an XServer/VNC session first, or run with X11 forwarding from Windows."
        )

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            args.browser_profile,
            headless=args.headless,
            accept_downloads=True,
        )
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        print(f"Opened {args.url}")
        print(f"Profile: {args.browser_profile}")
        print("Log in in the browser window, then press Enter here to close and save cookies.")
        input()
        context.close()


if __name__ == "__main__":
    main()
