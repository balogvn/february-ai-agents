from huggingface_hub import HfApi
import os

def push_files():
    api = HfApi()
    repo_id = "balogvn/swahili-ner-dataset"
    repo_path = "huggingface_dataset"

    api.upload_folder(
        folder_path=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="🤖 Auto-push new annotations"
    )
