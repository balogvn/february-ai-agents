from transformers import pipeline
import numpy as np

def ensure_python_type(value):
    """Convert numpy types to Python native types"""
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    elif isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    return value

def auto_annotate_text(texts):
    classifier = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    filtered_results = []

    for text in texts:
        ents = classifier(text)

        annotations = []
        for ent in ents:
            if ent["score"] >= 0.75:  # 🔍 QA filtering by confidence
                annotations.append({
                    "value": {
                        "start": ensure_python_type(ent["start"]),
                        "end": ensure_python_type(ent["end"]),
                        "labels": [ent["entity_group"]]
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "score": ensure_python_type(ent["score"])
                })

        if annotations:  # Only include if at least one high-confidence prediction
            filtered_results.append({
                "data": {"text": text},
                "predictions": [{"result": annotations}]
            })

    return filtered_results