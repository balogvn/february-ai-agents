import json

with open("annotations/swahili_export.json") as f:
    data = json.load(f)

for task in data:
    if "predictions" in task:
        for pred in task["predictions"]:
            pred["model_version"] = "v1"

with open("annotations/swahili_export_with_model_version.json", "w") as f:
    json.dump(data, f, indent=2)

