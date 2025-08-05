# 🇰🇪 Swahili Named Entity Recognition (NER) Dataset

This dataset contains automatically pre-annotated and human-reviewed Swahili sentences for Named Entity Recognition (NER) tasks.

Entities include:
- `PER` – Person
- `LOC` – Location
- `ORG` – Organization
- `MISC` – Miscellaneous

---

## 🧾 Dataset Info

- **Data Format:** JSON compatible with Label Studio
- **Model Used:** `dslim/bert-base-NER`
- **Annotations:** Auto-annotated + QA filtered
- **Source:** [February AI](https://github.com/kayodeb/february-ai-agents)
- **License:** MIT

---

## 🚀 Usage Example

```python
from datasets import load_dataset

dataset = load_dataset("balogvn/swahili-ner-dataset")
print(dataset["train"][0])


🙌 Contact
If you have questions, feedback, or want to collaborate:

📧 Email: balogvn@gmail.com
🐦 Twitter: @balogvn


📚 Citation

@misc{februaryai_swahili_ner,
  author = {February AI},
  title = {Swahili Named Entity Recognition (NER) Dataset},
  year = 2025,
  howpublished = {\url{https://huggingface.co/datasets/balogvn/swahili-ner-dataset}},
  note = {Automatically annotated and QA-reviewed Swahili dataset}
}


🔁 Updated automatically by the February AI pipeline.
