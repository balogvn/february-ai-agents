# orchestrator.py
import sys
import os
import json
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Allow import from current folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.auto_annotator_agent import auto_annotate_text
from scripts.push_to_huggingface import push_files
from scripts.generate_metadata import generate_metadata
from scripts.logger import log_info, log_error
from label_studio_sdk import Client
import requests

def notify_failure(error_message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    message = f"🚨 February AI Pipeline failed:\n\n{error_message}"
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={
        "chat_id": chat_id,
        "text": message
    })

def import_to_label_studio():
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    project_id = int(os.getenv("PROJECT_ID"))

    if not api_key or not url or not project_id:
        raise ValueError("❌ Missing Label Studio credentials")

    log_info(f"🔗 Connecting to Label Studio at {url}")
    ls = Client(url=url, api_key=api_key)
    project = ls.get_project(project_id)
    project.import_tasks("annotations/swahili_export.json")
    log_info("✅ Imported to Label Studio")

def run():
    try:
        # Set up
        file_path = "data/swahili_news.json"
        export_path = "annotations/swahili_export.json"
        metadata_path = "huggingface_dataset/metadata.json"

        log_info("🚀 Starting February AI pipeline")
        os.makedirs("annotations", exist_ok=True)
        os.makedirs("huggingface_dataset", exist_ok=True)

        # Load dataset
        with open(file_path) as f:
            data = json.load(f)
        texts = [item["text"] for item in data]
        log_info(f"📄 Loaded {len(texts)} texts")

        # Annotate
        annotations = auto_annotate_text(texts)

        # Save annotations
        with open(export_path, "w") as f:
            json.dump(annotations, f, indent=2)
        log_info(f"✅ Annotations saved to {export_path}")

        # Generate metadata
        generate_metadata(
            dataset_name="swahili-ner-dataset",
            num_samples=len(annotations),
            language="sw",
            model_used="dslim/bert-base-NER",
            output_path=metadata_path
        )
        log_info("📝 metadata.json generated")

        # Push to HF
        push_files()
        log_info("📤 Hugging Face push complete")

        # Load into Label Studio
        import_to_label_studio()

        log_info("✅ All tasks completed successfully")

    except Exception as e:
        log_error(f"❌ Pipeline failed: {e}")
        notify_failure(str(e))

if __name__ == "__main__":
    run()
