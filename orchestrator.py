# orchestrator.py
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.auto_annotator_agent import auto_annotate_text
from push_to_huggingface import push_files
from label_studio_sdk import Client

def run():
    # Load your dataset
    with open("data/swahili_news.json") as f:
        texts = [item["text"] for item in json.load(f)]

    # Annotate it
    annotations = auto_annotate_text(texts)

    # Save results
    os.makedirs("annotations", exist_ok=True)
    with open("annotations/swahili_export.json", "w") as f:
        json.dump(annotations, f, indent=2)

    # Push to HF
    push_files()

    # Load to Label Studio
    import_to_label_studio()

def import_to_label_studio():
    ls = Client(url="http://localhost:8080", api_key="YOUR_API_KEY")
    project = ls.get_project(YOUR_PROJECT_ID)
    project.import_tasks("annotations/swahili_export.json")

if __name__ == "__main__":
    run()
    print("\n✅ Orchestration finished. Check annotations/swahili_export.json")
