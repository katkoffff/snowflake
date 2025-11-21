# backend/utils/debug.py
import traceback
from datetime import datetime

def print_debug(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {msg}")

def log_error(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[ERROR {timestamp}] {msg}")
    traceback.print_exc()