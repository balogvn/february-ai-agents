import sys
import os
import json

# Add the directory of this script to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.auto_annotator_agent import auto_annotate_text

def run():
    # Load your dataset
    with open("data/swahili_news.json") as f:
        texts = [item["text"] for item in json.load(f)]

    # Annotate it
    annotations = auto_annotate_text(texts)

    # Make sure annotations directory exists
    os.makedirs("annotations", exist_ok=True)

    # Clear old output
    export_path = "annotations/swahili_export.json"
    if os.path.exists(export_path):
        os.remove(export_path)

    # Save results
    with open(export_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print("\n✅ Orchestration finished. Check annotations/swahili_export.json")

if __name__ == "__main__":
    run()
