# orchestrator.py
import sys
import os
import json
from dotenv import load_dotenv
from glob import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.auto_annotator_agent import auto_annotate_text
from push_to_huggingface import push_files
from label_studio_sdk import Client
from scripts.generate_metadata import generate_metadata

load_dotenv()

def run():
    os.makedirs("annotations", exist_ok=True)
    files = glob("data/*.json")

    for file_path in files:
        base_name = os.path.basename(file_path).replace(".json", "")
        export_path = f"annotations/{base_name}_annotated.json"

        # ✅ Skip if already annotated
        if os.path.exists(export_path):
            print(f"⏭️  Skipping {base_name} – already annotated.")
            continue

        print(f"🧠 Annotating: {file_path}")
        with open(file_path) as f:
            texts = [item["text"] for item in json.load(f)]

        annotations = auto_annotate_text(texts)

        with open(export_path, "w") as f:
            json.dump(annotations, f, indent=2)

        print(f"✅ Exported to {export_path}")

    # Metadata + Push + Label Studio
    generate_metadata()
    push_files()
    import_to_label_studio()

def import_to_label_studio():
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    project_id = int(os.getenv("PROJECT_ID"))

    if not api_key or not url or not project_id:
        raise ValueError("❌ LABEL STUDIO environment variables missing!")

    print(f"🔗 Connecting to Label Studio at {url}")
    print(f"📂 Using Project ID: {project_id}")

    ls = Client(url=url, api_key=api_key)
    project = ls.get_project(project_id)

    for file in glob("annotations/*_annotated.json"):
        project.import_tasks(file)
        print(f"📥 Imported {file} to Label Studio")

if __name__ == "__main__":
    run()
    print("\n✅ All orchestration finished. Annotated files stored in /annotations")
