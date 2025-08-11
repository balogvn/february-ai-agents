import os
from huggingface_hub import HfApi, snapshot_download

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

api = HfApi()
search_term = "swahili"

print(f"🔍 Searching Hugging Face for '{search_term}' datasets...")
datasets = api.list_datasets(search=search_term)

found = 0
downloaded = 0

for ds in datasets:
    language = []
    tags = []

    if ds.cardData:
        language = [lang.lower() for lang in ds.cardData.get("language", [])]
        tags = [tag.lower() for tag in ds.cardData.get("tags", [])]

    # Match if swahili appears in language, tags, or dataset name
    if (search_term.lower() in language) or \
       (search_term.lower() in tags) or \
       (search_term.lower() in ds.id.lower()):
        found += 1
        print(f"📥 Downloading: {ds.id}")
        try:
            snapshot_download(
                repo_id=ds.id,
                repo_type="dataset",
                local_dir=os.path.join(DATA_DIR, ds.id.replace("/", "_")),
                local_dir_use_symlinks=False
            )
            downloaded += 1
        except Exception as e:
            print(f"❌ Failed to download {ds.id}: {e}")

print(f"\n✅ Found {found} datasets, successfully downloaded {downloaded}.")
print(f"📂 All datasets are in: {os.path.abspath(DATA_DIR)}")
