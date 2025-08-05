# orchestrator.py
import sys
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure this script's directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.auto_annotator_agent import auto_annotate_text
from scripts.generate_metadata import generate_metadata
from push_to_huggingface import push_files
from label_studio_sdk import Client

def run():
    # Load your dataset
    with open("data/swahili_news.json") as f:
        texts = [item["text"] for item in json.load(f)]

    # Annotate
    annotations = auto_annotate_text(texts)

    # Save results
    os.makedirs("annotations", exist_ok=True)
    output_path = "annotations/swahili_export.json"
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2)

    # ✅ Generate metadata
    generate_metadata(output_path)

    # ✅ Push to HF
    push_files()

    # ✅ Load to Label Studio
    import_to_label_studio()

def import_to_label_studio():
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    project_id = int(os.getenv("PROJECT_ID", 0))

    if not api_key or not url or not project_id:
        raise ValueError("❌ Missing Label Studio credentials in .env")

    print(f"🔗 Connecting to Label Studio at {url}")
    print(f"📂 Using Project ID: {project_id}")

    ls = Client(url=url, api_key=api_key)
    project = ls.get_project(project_id)
    project.import_tasks("annotations/swahili_export.json")

    print("✅ Successfully imported to Label Studio!")

if __name__ == "__main__":
    run()
    print("\n✅ Orchestration finished. Check annotations/swahili_export.json and metadata.json")
