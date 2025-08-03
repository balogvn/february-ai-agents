import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from agents.auto_annotator_agent import auto_annotate_text
import json

def run():
    # Load your dataset
    with open("data/swahili_news.json") as f:
        texts = [item["text"] for item in json.load(f)]

    # Annotate it
    annotations = auto_annotate_text(texts)

    # Save results
    with open("annotations/swahili_export.json", "w") as f:
        json.dump(annotations, f, indent=2)

if __name__ == "__main__":
    run()
