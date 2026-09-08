# -*- coding: utf-8 -*-
"""
Streamlit Web UI Launcher
=========================
Launches the OpenAgriNet (OAN) Kenya Pest & Disease Diagnostic Web Application.

Usage:
    python scripts/run_ui.py
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(REPO_ROOT, "ui", "app.py")

def main():
    print("=" * 80)
    print("  LAUNCHING OPENAGRINET (OAN) KENYA PEST & DISEASE DIAGNOSTIC INTERFACE")
    print("=" * 80)
    print(f"App Path: {APP_PATH}")
    print("Starting local Streamlit server at: http://localhost:8501")
    print("Press Ctrl+C to terminate the web server.")
    print("=" * 80)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        APP_PATH,
        "--browser.gatherUsageStats",
        "false",
        "--server.port",
        "8501",
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nWeb server terminated cleanly.")

if __name__ == "__main__":
    main()
