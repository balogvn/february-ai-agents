import json
import os
from datetime import datetime

def generate_metadata(dataset_path, model_name="dslim/bert-base-NER", language="sw"):
    with open(dataset_path) as f:
        data = json.load(f)

    metadata = {
        "dataset_name": os.path.basename(dataset_path),
        "num_samples": len(data),
        "language": language,
        "model_used": model_name,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    metadata_path = os.path.join(os.path.dirname(dataset_path), "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata written to {metadata_path}")
