# scripts/generate_metadata.py

import json
import os
from datetime import datetime

def generate_metadata(dataset_name, num_samples, language="sw", model_used="dslim/bert-base-NER"):
    metadata = {
        "dataset_name": dataset_name,
        "num_samples": num_samples,
        "language": language,
        "model_used": model_used,
       "generated_at": datetime.utcnow().isoformat() + "Z"

    }

    os.makedirs("annotations", exist_ok=True)
    metadata_path = "annotations/metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata written to {metadata_path}")
