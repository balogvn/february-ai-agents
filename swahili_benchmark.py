import json
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
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
    tokenized = tokenizer(text, return_offsets_mapping=True, truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
    offsets = tokenized["offset_mapping"]

    # Initialize labels
    true_char_labels = ["O"] * len(text)
    pred_char_labels = ["O"] * len(text)

    # Get TRUE labels from annotations
    if "annotations" in item:
        for ann in item["annotations"][0]["result"]:
            if ann["type"] != "labels":
                continue
            start = ann["value"]["start"]
            end = ann["value"]["end"]
            label = ann["value"]["labels"][0]
            true_char_labels[start] = f"B-{label}"
            for i in range(start + 1, end):
                true_char_labels[i] = f"I-{label}"

    # Get PREDICTED labels
    for pred in item["predictions"][0]["result"]:
        if pred["type"] != "labels":
            continue
        start = pred["value"]["start"]
        end = pred["value"]["end"]
        label = pred["value"]["labels"][0]
        pred_char_labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            pred_char_labels[i] = f"I-{label}"

    # Align with tokens
    aligned_true = []
    aligned_pred = []
    for start, end in offsets:
        if end == 0:
            continue
        aligned_true.append(true_char_labels[start] if start < len(true_char_labels) else "O")
        aligned_pred.append(pred_char_labels[start] if start < len(pred_char_labels) else "O")

    all_true_labels.append(aligned_true)
    all_predictions.append(aligned_pred)

# Evaluate
print("📊 Classification Report")
print(classification_report(all_true_labels, all_predictions))
print("Precision: {:.1f}%".format(100 * precision_score(all_true_labels, all_predictions)))
print("Recall:    {:.1f}%".format(100 * recall_score(all_true_labels, all_predictions)))
print("F1 Score:  {:.1f}%".format(100 * f1_score(all_true_labels, all_predictions)))
