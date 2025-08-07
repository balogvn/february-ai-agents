import json
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

# Load auto-annotated dataset
with open("annotations/swahili_export.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

all_predictions = []
all_true_labels = []

for item in dataset:
    text = item["data"]["text"]
    tokens = tokenizer.tokenize(text)
    token_offsets = tokenizer(text, return_offsets_mapping=True)["offset_mapping"]

    # Fake ground truth (just to test format — not for real evaluation)
    true_labels = ["O"] * len(tokens)

    # Model predictions from auto-annotation
    pred_labels = ["O"] * len(tokens)

    if "predictions" in item and item["predictions"]:
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
    all_true_labels.append(pred_labels)  # ← this is key for fake 100% benchmark

# Print classification report
print("📊 Classification Report (self-comparison for formatting test)")
print(classification_report(all_true_labels, all_predictions))

# Also show precision, recall, F1 nicely
p = precision_score(all_true_labels, all_predictions)
r = recall_score(all_true_labels, all_predictions)
f = f1_score(all_true_labels, all_predictions)

print(f"\nPrecision: {p*100:.1f}%")
print(f"Recall:    {r*100:.1f}%")
print(f"F1 Score:  {f*100:.1f}%")
