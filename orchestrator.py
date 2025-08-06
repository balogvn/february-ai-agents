# orchestrator.py
import sys
import os
import json
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

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

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
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
    elif hasattr(obj, 'item'):  # Handle single numpy scalars
        return obj.item()
    else:
        return obj

def send_telegram_message(message, is_success=False):
    """Send a message via Telegram"""
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
                "parse_mode": "HTML"  # Allow HTML formatting
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
    """Send failure notification"""
    message = f"🚨 <b>February AI Pipeline FAILED</b>\n\n❌ Error: {error_message}\n\n🕐 Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    send_telegram_message(message, is_success=False)

def notify_success(stats):
    """Send success notification with stats"""
    message = f"""🎉 <b>February AI Pipeline SUCCESS!</b>

✅ <b>Pipeline completed successfully</b>

📊 <b>Stats:</b>
• Texts processed: {stats.get('texts_processed', 'N/A')}
• Annotations created: {stats.get('annotations_created', 'N/A')}
• HuggingFace upload: {'✅ Success' if stats.get('hf_success') else '❌ Failed'}
• Label Studio import: {'✅ Success' if stats.get('ls_success') else '⚠️ Skipped'}

🕐 <b>Completed:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC

🔗 Check your Hugging Face repository for the latest dataset!"""
    
    send_telegram_message(message, is_success=True)

def import_to_label_studio():
    # Get credentials from environment variables (GitHub Secrets)
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    project_id = os.getenv("PROJECT_ID")

    # Check if all credentials are available
    if not api_key:
        log_info("⚠️ LABEL_STUDIO_API_KEY not found - skipping Label Studio import")
        return False
    
    if not url:
        log_info("⚠️ LABEL_STUDIO_URL not found - skipping Label Studio import")
        return False
        
    if not project_id:
        log_info("⚠️ PROJECT_ID not found - skipping Label Studio import")
        return False

    try:
        project_id = int(project_id)
        log_info(f"🔗 Connecting to Label Studio")
        log_info(f"📂 Using Project ID: {project_id}")
        
        ls = Client(url=url, api_key=api_key)
        project = ls.get_project(project_id)
        project.import_tasks("annotations/swahili_export.json")
        log_info("✅ Successfully imported to Label Studio")
        return True
        
    except ValueError as e:
        log_error(f"❌ Invalid PROJECT_ID format: {e}")
        return False
    except Exception as e:
        log_error(f"❌ Label Studio import failed: {e}")
        log_info("Continuing pipeline without Label Studio import...")
        return False

def run():
    # Initialize stats for success notification
    stats = {
        'texts_processed': 0,
        'annotations_created': 0,
        'hf_success': False,
        'ls_success': False
    }
    
    try:
        # Set up
        file_path = "data/swahili_news.json"
        export_path = "annotations/swahili_export.json"

        log_info("🚀 Starting February AI pipeline")
        os.makedirs("annotations", exist_ok=True)
        os.makedirs("huggingface_dataset", exist_ok=True)

        # Load dataset
        with open(file_path) as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        stats['texts_processed'] = len(texts)
        log_info(f"📄 Loaded {len(texts)} texts")

        # Annotate
        log_info("🤖 Starting auto-annotation process...")
        annotations = auto_annotate_text(texts)
        log_info("✅ Auto-annotation completed")

        # Convert numpy types to JSON-serializable types
        log_info("🔄 Converting data types for JSON serialization...")
        annotations_clean = convert_numpy_types(annotations)
        stats['annotations_created'] = len(annotations_clean) if isinstance(annotations_clean, list) else 1

        # Save annotations with proper encoding and fallback serialization
        with open(export_path, "w", encoding='utf-8') as f:
            json.dump(annotations_clean, f, indent=2, ensure_ascii=False, default=str)
        log_info(f"✅ Annotations saved to {export_path}")

        # Generate metadata
        log_info("📝 Generating metadata...")
        generate_dataset_metadata(
            dataset_name="swahili-ner-dataset",
            num_samples=stats['annotations_created'],
            language="sw",
            model_used="dslim/bert-base-NER"
        )
        log_info("✅ Metadata generation completed")

        # Push to HF
        log_info("📤 Starting Hugging Face upload...")
        try:
            push_files()
            stats['hf_success'] = True
            log_info("✅ Hugging Face push complete")
        except Exception as e:
            log_error(f"❌ Hugging Face push failed: {e}")
            stats['hf_success'] = False

        # Load into Label Studio (with graceful failure handling)
        log_info("📥 Starting Label Studio import...")
        stats['ls_success'] = import_to_label_studio()

        log_info("🎉 All tasks completed successfully!")
        
        # Send success notification
        notify_success(stats)

    except FileNotFoundError as e:
        error_msg = f"File not found: {e}"
        log_error(f"❌ {error_msg}")
        notify_failure(error_msg)
        raise
    except json.JSONDecodeError as e:
        error_msg = f"JSON parsing error: {e}"
        log_error(f"❌ {error_msg}")
        notify_failure(error_msg)
        raise
    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        log_error(f"❌ {error_msg}")
        notify_failure(error_msg)
        raise

if __name__ == "__main__":
    run()