# scripts/logger.py
import logging
from datetime import datetime
import os

# Set up logs folder
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_info(message):
    print("ℹ️", message)
    logging.info(message)

def log_error(message):
    print("❌", message)
    logging.error(message)
