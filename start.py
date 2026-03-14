import subprocess
import sys
import os
import time
import threading

def run_api():
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ])

def run_ui():
    # Wait for API to start first
    time.sleep(5)
    subprocess.run([
        sys.executable, "ui/app.py"
    ])

if __name__ == "__main__":
    # Run both in parallel threads
    api_thread = threading.Thread(target=run_api)
    ui_thread = threading.Thread(target=run_ui)

    api_thread.start()
    ui_thread.start()

    api_thread.join()
    ui_thread.join()