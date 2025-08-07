import json
from seqeval.metrics import classification_report
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Load annotated dataset
with open("annotations/swahili_export.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

all_predictions = []
all_true_labels = []

for item in dataset:
    text = item["data"]["text"]
    tokens = tokenizer.tokenize(text)
    token_offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]

    # Ground truth placeholder (if available)
    true_labels = ["O"] * len(tokens)

    # Prediction labels
    pred_labels = ["O"] * len(tokens)

    entities = item["predictions"][0]["result"]

    for entity in entities:
        label = entity["value"]["labels"][0]
        start_char = entity["value"]["start"]
        end_char = entity["value"]["end"]

        for i, (start, end) in enumerate(token_offsets):
            if start >= end_char:
                break
            if end <= start_char:
                continue
            if start >= start_char and end <= end_char:
                if start == start_char:
                    pred_labels[i] = f"B-{label}"
                else:
                    pred_labels[i] = f"I-{label}"

    all_predictions.append(pred_labels)
    all_true_labels.append(true_labels)  # Placeholder since we don’t have gold labels

# Since we don't have gold labels, we print only predicted labels
for preds in all_predictions:
    print(preds)

# If you later add gold labels, uncomment this:
# print(classification_report(all_true_labels,_
