from transformers import pipeline

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
                        "start": ent["start"],
                        "end": ent["end"],
                        "labels": [ent["entity_group"]]
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "score": ent["score"]  # Optional: include score
                })

        if annotations:  # Only include if at least one high-confidence prediction
            filtered_results.append({
                "data": {"text": text},
                "predictions": [{"result": annotations}]
            })

    return filtered_results
