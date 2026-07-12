"""Visit the Streamlit app in a real browser so Community Cloud counts it as
viewer traffic, and click the wake-up button if the app has gone to sleep.

Plain HTTP pings don't work: a sleeping app still returns HTTP 200 with a
static shell, so an actual browser session (websocket) is required.

Usage: python keep_alive.py <app_url>
"""

import sys

from playwright.sync_api import sync_playwright

WAKE_BUTTON_TEXT = "get this app back up"


def main() -> None:
    app_url = sys.argv[1]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(app_url, wait_until="load", timeout=120_000)

        # Let the page settle so the sleeping screen (if any) has rendered.
        page.wait_for_timeout(10_000)

        wake_button = page.get_by_text(WAKE_BUTTON_TEXT, exact=False)
        if wake_button.count() > 0:
            print("App is asleep — clicking the wake-up button.")
            wake_button.first.click()
            # Give the container time to boot before disconnecting.
            page.wait_for_timeout(120_000)
        else:
            print("App appears to be awake.")

        # Stay connected briefly so the visit registers as viewer traffic.
        page.wait_for_timeout(30_000)
        browser.close()

    print(f"Finished visiting {app_url}")


if __name__ == "__main__":
    main()
