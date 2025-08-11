import sys
import os
import json
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from glob import glob
from pathlib import Path
import hashlib

# Load environment variables from .env file
load_dotenv()

# Allow import from current folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.auto_annotator_agent import auto_annotate_text
from scripts.push_to_huggingface import push_files
from scripts.generate_metadata import generate_dataset_metadata
from scripts.logger import log_info, log_error
from label_studio_sdk import Client
import requests

# Skip tracking file
SKIP_LOG_FILE = "processed_files.log"

def convert_numpy_types(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def get_file_hash(file_path):
    """Generate a hash for a file to track if it's been processed"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        log_error(f"Failed to generate hash for {file_path}: {e}")
        return None

def load_skip_log():
    """Load the list of previously processed files"""
    if os.path.exists(SKIP_LOG_FILE):
        try:
            with open(SKIP_LOG_FILE, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            log_error(f"Failed to load skip log: {e}")
    return set()

def update_skip_log(file_hash):
    """Add a file hash to the skip log"""
    try:
        with open(SKIP_LOG_FILE, 'a') as f:
            f.write(f"{file_hash}\n")
    except Exception as e:
        log_error(f"Failed to update skip log: {e}")

def clear_skip_log():
    """Clear the skip log file"""
    try:
        if os.path.exists(SKIP_LOG_FILE):
            os.remove(SKIP_LOG_FILE)
            log_info("✅ Skip log cleared - all files will be reprocessed")
        else:
            log_info("ℹ️ No skip log found to clear")
    except Exception as e:
        log_error(f"Failed to clear skip log: {e}")

def find_dataset_files(data_dir="data"):
    """Recursively find all .json and .jsonl files in the data directory"""
    data_path = Path(data_dir)
    if not data_path.exists():
        log_error(f"Data directory '{data_dir}' does not exist")
        return []
    
    files = []
    for pattern in ["**/*.json", "**/*.jsonl"]:
        files.extend(data_path.glob(pattern))
    
    log_info(f"📁 Found {len(files)} dataset files in '{data_dir}'")
    return [str(f) for f in files]

def load_dataset_file(file_path):
    """Load data from either JSON or JSONL file"""
    file_extension = Path(file_path).suffix.lower()
    texts = []
    
    try:
        if file_extension == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of objects with "text" field
                texts = [item.get("text", str(item)) for item in data if isinstance(item, dict)]
                # If no "text" field found, try other common fields
                if not texts and data:
                    for field in ["content", "message", "body", "description"]:
                        texts = [item.get(field) for item in data if isinstance(item, dict) and item.get(field)]
                        if texts:
                            break
                # If still no texts, treat each item as text
                if not texts:
                    texts = [str(item) for item in data]
            elif isinstance(data, dict):
                # Single object or nested structure
                if "text" in data:
                    texts = [data["text"]]
                elif "data" in data and isinstance(data["data"], list):
                    texts = [item.get("text", str(item)) for item in data["data"]]
                else:
                    texts = [str(data)]
                    
        elif file_extension == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                text = obj.get("text") or obj.get("content") or obj.get("message") or str(obj)
                            else:
                                text = str(obj)
                            texts.append(text)
                        except json.JSONDecodeError:
                            # If line is not valid JSON, treat as plain text
                            texts.append(line)
        
        # Filter out None and empty strings
        texts = [text for text in texts if text and str(text).strip()]
        
    except Exception as e:
        log_error(f"Failed to load {file_path}: {e}")
        return []
    
    return texts

def send_telegram_message(message, is_success=False):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        log_info("⚠️ Telegram credentials missing - skipping notification")
        return False
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage", 
            data={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
        )
        if response.status_code == 200:
            emoji = "✅" if is_success else "📱"
            log_info(f"{emoji} Telegram notification sent successfully")
            return True
        else:
            log_error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        log_error(f"Failed to send Telegram notification: {e}")
        return False

def notify_failure(error_message):
    message = f"🚨 <b>February AI Pipeline FAILED</b>\n\n❌ Error: {error_message}\n\n🕐 Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    send_telegram_message(message, is_success=False)

def notify_success(stats):
    message = f"""🎉 <b>February AI Pipeline SUCCESS!</b>

✅ <b>Pipeline completed successfully</b>

📊 <b>Stats:</b>
• Files processed: {stats.get('files_processed', 'N/A')}
• Files skipped: {stats.get('files_skipped', 'N/A')}
• Texts processed: {stats.get('texts_processed', 'N/A')}
• Annotations created: {stats.get('annotations_created', 'N/A')}
• HuggingFace upload: {'✅ Success' if stats.get('hf_success') else '❌ Failed'}
• Label Studio import: {'✅ Success' if stats.get('ls_success') else '⚠️ Skipped'}

🔍 <b>Check Hugging Face</b> for your updated dataset!

🕐 <b>Completed:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"""
    send_telegram_message(message, is_success=True)

def import_to_label_studio():
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    project_id = os.getenv("PROJECT_ID")

    if not api_key or not url or not project_id:
        log_info("⚠️ Missing Label Studio credentials - skipping import")
        return False

    try:
        project_id = int(project_id)
        log_info(f"🔗 Connecting to Label Studio Project {project_id}")
        ls = Client(url=url, api_key=api_key)
        project = ls.get_project(project_id)
        project.import_tasks("annotations/all_swahili_annotations.json")
        log_info("✅ Successfully imported to Label Studio")
        return True
    except Exception as e:
        log_error(f"❌ Label Studio import failed: {e}")
        return False

def run(clear_skip=False):
    """
    Main pipeline function
    
    Args:
        clear_skip (bool): If True, clear the skip log and reprocess all files
    """
    stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'texts_processed': 0,
        'annotations_created': 0,
        'hf_success': False,
        'ls_success': False
    }

    try:
        log_info("🚀 Starting February AI pipeline")
        
        # Clear skip log if requested
        if clear_skip:
            clear_skip_log()
        
        # Create necessary directories
        os.makedirs("annotations", exist_ok=True)
        os.makedirs("huggingface_dataset", exist_ok=True)

        # Load skip log
        processed_files = load_skip_log()
        log_info(f"📋 Loaded skip log with {len(processed_files)} previously processed files")

        all_annotations = []
        all_texts = []

        # Find all dataset files recursively
        dataset_files = find_dataset_files()
        
        if not dataset_files:
            log_error("❌ No dataset files found in data/ directory")
            return

        # Process each file
        for file_path in dataset_files:
            # Check if file was already processed
            file_hash = get_file_hash(file_path)
            if file_hash and file_hash in processed_files:
                log_info(f"⏭️ Skipping already processed file: {file_path}")
                stats['files_skipped'] += 1
                continue

            log_info(f"📂 Processing file: {file_path}")
            
            # Load texts from file
            texts = load_dataset_file(file_path)
            
            if not texts:
                log_info(f"⚠️ No texts found in {file_path}")
                continue
                
            stats['texts_processed'] += len(texts)
            all_texts.extend(texts)

            log_info(f"🤖 Annotating {len(texts)} texts from {os.path.basename(file_path)}")
            annotations = auto_annotate_text(texts)
            annotations_clean = convert_numpy_types(annotations)
            all_annotations.extend(annotations_clean)
            
            # Mark file as processed
            if file_hash:
                update_skip_log(file_hash)
            
            stats['files_processed'] += 1

        stats['annotations_created'] = len(all_annotations)
        
        if not all_annotations:
            log_info("⚠️ No annotations created - nothing to save")
            return

        # Save combined annotations
        export_path = "annotations/all_swahili_annotations.json"
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(all_annotations, f, indent=2, ensure_ascii=False)
        log_info(f"✅ All annotations saved to {export_path}")

        # Generate metadata
        generate_dataset_metadata(
            dataset_name="swahili-ner-dataset",
            num_samples=stats['annotations_created'],
            language="sw",
            model_used="dslim/bert-base-NER"
        )

        # Push to Hugging Face
        try:
            push_files()
            stats['hf_success'] = True
            log_info("✅ Hugging Face push complete")
        except Exception as e:
            log_error(f"❌ Hugging Face push failed: {e}")

        # Import to Label Studio
        stats['ls_success'] = import_to_label_studio()

        log_info("🎉 Pipeline completed successfully!")
        log_info("🔍 **Check Hugging Face** for your updated dataset!")
        notify_success(stats)

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        log_error(f"❌ {error_msg}")
        notify_failure(error_msg)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="February AI Pipeline")
    parser.add_argument("--clear-skip", action="store_true", 
                       help="Clear the skip log and reprocess all files")
    args = parser.parse_args()
    
    run(clear_skip=args.clear_skip)