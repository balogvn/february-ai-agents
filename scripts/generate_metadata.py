# scripts/generate_metadata.py

import json
import os
from datetime import datetime

def generate_dataset_metadata(dataset_name, num_samples, language="sw", model_used="dslim/bert-base-NER"):
    """Generate metadata for the dataset"""
    metadata = {
        "dataset_name": dataset_name,
        "num_samples": num_samples,
        "language": language,
        "model_used": model_used,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    # Create the directory if it doesn't exist
    os.makedirs("huggingface_dataset", exist_ok=True)
    metadata_path = "huggingface_dataset/metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata written to {metadata_path}")
    return metadata_path