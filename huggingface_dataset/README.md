# Swahili Named Entity Recognition (NER) Dataset

This dataset contains automatically pre-annotated and human-reviewed Swahili sentences for NER tasks.
Annotations include: PER (Person), LOC (Location), ORG (Organization), MISC.

Data Format: JSON compatible with Label Studio  
Source: February AI (https://github.com/kayodeb/february-ai-agents)
License: MIT


## Usage Example

```python
from datasets import load_dataset

dataset = load_dataset("balogvn/swahili-ner-dataset")
print(dataset["train"][0])

```bash
cd swahili-ner-dataset
git add README.md
git commit -m "📚 Add sample usage code"
git push

