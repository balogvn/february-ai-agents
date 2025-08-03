from transformers import pipeline

def auto_annotate_text(texts):
    classifier = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    
    results = []

    for text in texts:
        ents = classifier(text)
        annotations = [
            {
                "value": {
                    "start": ent["start"],
                    "end": ent["end"],
                    "labels": [ent["entity_group"]]
                },
                "from_name": "label",
                "to_name": "text",
                "type": "labels"
            } for ent in ents
        ]
        results.append({
            "data": {"text": text},
            "annotations": [{"result": annotations}]
        })

    return results
