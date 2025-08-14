import sys
import os
import json
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from glob import glob
from pathlib import Path
import hashlib
import time
from typing import List, Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Allow import from current folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.auto_annotator_agent import auto_annotate_text
from scripts.push_to_huggingface import push_files
from scripts.logger import log_info, log_error
from label_studio_sdk import Client
import requests

# Skip tracking file
SKIP_LOG_FILE = "processed_files.log"

# Default limits - can be overridden
DEFAULT_MAX_FILES = 10
DEFAULT_MAX_TEXTS_PER_FILE = 50
DEFAULT_MAX_TOTAL_TEXTS = 500
DEFAULT_MAX_TEXT_LENGTH = 5000  # characters

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj

def get_file_hash(file_path):
    """Generate a hash for a file to track if it's been processed"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        log_error(f"Failed to generate hash for {file_path}: {e}")
        return None

def get_file_size_mb(file_path):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0

def load_skip_log():
    """Load the list of previously processed files"""
    if os.path.exists(SKIP_LOG_FILE):
        try:
            with open(SKIP_LOG_FILE, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        except Exception as e:
            log_error(f"Failed to load skip log: {e}")
    return set()

def update_skip_log(file_hash):
    """Add a file hash to the skip log"""
    try:
        with open(SKIP_LOG_FILE, 'a') as f:
            f.write(f"{file_hash}\n")
    except Exception as e:
        log_error(f"Failed to update skip log: {e}")

def clear_skip_log():
    """Clear the skip log file"""
    try:
        if os.path.exists(SKIP_LOG_FILE):
            os.remove(SKIP_LOG_FILE)
            log_info("✅ Skip log cleared - all files will be reprocessed")
        else:
            log_info("ℹ️ No skip log found to clear")
    except Exception as e:
        log_error(f"Failed to clear skip log: {e}")

def find_dataset_files(data_dir="data", max_files=None, max_file_size_mb=None):
    """Recursively find and limit dataset files"""
    data_path = Path(data_dir)
    if not data_path.exists():
        log_error(f"Data directory '{data_dir}' does not exist")
        return []
    
    all_files = []
    for pattern in ["**/*.json", "**/*.jsonl", "**/*.txt"]:
        all_files.extend(data_path.glob(pattern))
    
    # Filter by file size if specified
    if max_file_size_mb:
        filtered_files = []
        for file_path in all_files:
            size_mb = get_file_size_mb(file_path)
            if size_mb <= max_file_size_mb:
                filtered_files.append(file_path)
            else:
                log_info(f"⚠️ Skipping large file ({size_mb:.1f}MB): {file_path}")
        all_files = filtered_files
    
    # Sort by file size (smallest first for faster processing)
    all_files.sort(key=lambda f: get_file_size_mb(f))
    
    # Limit number of files
    if max_files and len(all_files) > max_files:
        log_info(f"🔢 Limiting to first {max_files} files (out of {len(all_files)} found)")
        all_files = all_files[:max_files]
    
    # Sort files by type for better logging
    json_files = [f for f in all_files if f.suffix.lower() in ['.json', '.jsonl']]
    txt_files = [f for f in all_files if f.suffix.lower() == '.txt']
    
    log_info(f"📁 Processing {len(all_files)} dataset files from '{data_dir}':")
    log_info(f"   • {len(json_files)} JSON/JSONL files")
    log_info(f"   • {len(txt_files)} TXT files")
    
    if max_file_size_mb:
        log_info(f"   • Max file size: {max_file_size_mb}MB")
    
    return [str(f) for f in all_files]

def truncate_text(text, max_length):
    """Truncate text to maximum length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def load_dataset_file(file_path, max_texts=None, max_text_length=None):
    """Load data from JSON, JSONL, or TXT file with limits"""
    file_extension = Path(file_path).suffix.lower()
    texts = []
    start_time = time.time()
    
    try:
        if file_extension == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of objects with "text" field
                texts = [item.get("text", str(item)) for item in data if isinstance(item, dict)]
                # If no "text" field found, try other common fields
                if not texts and data:
                    for field in ["content", "message", "body", "description"]:
                        texts = [item.get(field) for item in data if isinstance(item, dict) and item.get(field)]
                        if texts:
                            break
                # If still no texts, treat each item as text
                if not texts:
                    texts = [str(item) for item in data]
            elif isinstance(data, dict):
                # Single object or nested structure
                if "text" in data:
                    texts = [data["text"]]
                elif "data" in data and isinstance(data["data"], list):
                    texts = [item.get("text", str(item)) for item in data["data"]]
                else:
                    texts = [str(data)]
                    
        elif file_extension == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    if max_texts and line_count >= max_texts:
                        log_info(f"🔢 Limiting JSONL to first {max_texts} lines")
                        break
                    
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                text = obj.get("text") or obj.get("content") or obj.get("message") or str(obj)
                            else:
                                text = str(obj)
                            texts.append(text)
                            line_count += 1
                        except json.JSONDecodeError:
                            # If line is not valid JSON, treat as plain text
                            texts.append(line)
                            line_count += 1
                            
        elif file_extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                
                if not content:
                    log_info(f"⚠️ Empty TXT file: {file_path}")
                    return []
                
                # Split text into chunks for processing
                # Option 1: Split by paragraphs (double newlines)
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                
                if paragraphs and len(paragraphs) > 1:
                    texts = paragraphs
                    log_info(f"📄 Split TXT file into {len(texts)} paragraphs")
                else:
                    # Option 2: Split by single newlines
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    if len(lines) > 1:
                        texts = lines
                        log_info(f"📄 Split TXT file into {len(texts)} lines")
                    else:
                        # Option 3: Treat entire content as single text
                        texts = [content]
                        log_info(f"📄 Processing TXT file as single text block")
        
        # Filter out None and empty strings
        texts = [text for text in texts if text and str(text).strip()]
        
        # Apply text limits
        original_count = len(texts)
        
        # Limit number of texts per file
        if max_texts and len(texts) > max_texts:
            texts = texts[:max_texts]
            log_info(f"🔢 Limited to first {max_texts} texts (was {original_count})")
        
        # Filter very short texts (less than 10 characters)
        texts = [text for text in texts if len(str(text).strip()) >= 10]
        
        # Truncate long texts
        if max_text_length:
            truncated_count = 0
            for i, text in enumerate(texts):
                if len(text) > max_text_length:
                    texts[i] = truncate_text(text, max_text_length)
                    truncated_count += 1
            
            if truncated_count > 0:
                log_info(f"✂️ Truncated {truncated_count} texts to {max_text_length} characters")
        
        # Final filtering
        short_filtered = original_count - len([t for t in texts if len(str(t).strip()) >= 10])
        if short_filtered > 0:
            log_info(f"⚠️ Filtered out {short_filtered} texts shorter than 10 characters")
        
        processing_time = time.time() - start_time
        log_info(f"⏱️ Processed {file_path} in {processing_time:.2f}s - {len(texts)} texts extracted")
        
    except Exception as e:
        log_error(f"Failed to load {file_path}: {e}")
        return []
    
    return texts

def process_in_batches(texts: List[str], batch_size: int = 20) -> List:
    """Process texts in smaller batches to manage memory and time"""
    all_annotations = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    log_info(f"🔄 Processing {len(texts)} texts in {total_batches} batches of {batch_size}")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        log_info(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
        
        start_time = time.time()
        try:
            batch_annotations = auto_annotate_text(batch_texts)
            batch_annotations_clean = convert_numpy_types(batch_annotations)
            all_annotations.extend(batch_annotations_clean)
            
            processing_time = time.time() - start_time
            log_info(f"✅ Batch {batch_num} completed in {processing_time:.2f}s")
            
        except Exception as e:
            log_error(f"❌ Failed to process batch {batch_num}: {e}")
            # Continue with next batch instead of failing completely
            continue
    
    return all_annotations

def safe_json_write(file_path: str, data: Any, ensure_ascii: bool = False) -> bool:
    """Safely write JSON data to file with error handling"""
    try:
        # Convert numpy types before writing
        clean_data = convert_numpy_types(data)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, ensure_ascii=ensure_ascii, separators=(',', ':'))
        
        log_info(f"✅ Successfully wrote {file_path}")
        return True
    except Exception as e:
        log_error(f"❌ Failed to write {file_path}: {e}")
        return False

def safe_jsonl_write(file_path: str, data: List[Dict], ensure_ascii: bool = False) -> bool:
    """Safely write JSONL data to file with error handling"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in data:
                # Convert numpy types before writing each record
                clean_record = convert_numpy_types(record)
                json.dump(clean_record, f, ensure_ascii=ensure_ascii, separators=(',', ':'))
                f.write('\n')
        
        log_info(f"✅ Successfully wrote {len(data)} records to {file_path}")
        return True
    except Exception as e:
        log_error(f"❌ Failed to write {file_path}: {e}")
        return False

def create_standardized_record(annotation: Dict, record_id: int) -> Dict[str, Any]:
    """Create a standardized record format for HuggingFace compatibility"""
    try:
        # Extract text safely
        text = ""
        if 'data' in annotation and 'text' in annotation['data']:
            text = str(annotation['data']['text']).strip()
        elif 'data' in annotation and isinstance(annotation['data'], str):
            text = str(annotation['data']).strip()
        else:
            log_error(f"No text found in annotation {record_id}")
            return None
        
        if not text:
            log_error(f"Empty text in annotation {record_id}")
            return None
        
        # Split text into tokens (simple whitespace splitting)
        tokens = text.split()
        if not tokens:
            log_error(f"No tokens found for text in annotation {record_id}")
            return None
        
        # Initialize all labels as 'O' (Outside)
        labels = ['O'] * len(tokens)
        entities = []
        
        # Extract entities from predictions
        if 'predictions' in annotation and annotation['predictions']:
            prediction = annotation['predictions'][0]
            
            if 'result' in prediction and prediction['result']:
                for result_item in prediction['result']:
                    if result_item.get('type') == 'labels' and 'value' in result_item:
                        try:
                            entity_start = int(result_item['value'].get('start', 0))
                            entity_end = int(result_item['value'].get('end', 0))
                            entity_text = str(result_item['value'].get('text', '')).strip()
                            entity_labels = result_item['value'].get('labels', [])
                            
                            if entity_labels and entity_text and entity_start < entity_end:
                                entity_label = str(entity_labels[0])
                                
                                # Find which tokens correspond to this entity
                                current_pos = 0
                                for token_idx, token in enumerate(tokens):
                                    token_start = text.find(token, current_pos)
                                    token_end = token_start + len(token)
                                    
                                    # Check if this token overlaps with the entity
                                    if (token_start < entity_end and token_end > entity_start):
                                        # Determine if this is the beginning or inside of entity
                                        is_beginning = True
                                        # Check if previous token was also part of this entity
                                        if token_idx > 0 and labels[token_idx - 1].endswith(f"-{entity_label}"):
                                            is_beginning = False
                                        
                                        labels[token_idx] = f"{'B' if is_beginning else 'I'}-{entity_label}"
                                    
                                    current_pos = token_end
                                
                                # Add entity info
                                entities.append({
                                    "start": entity_start,
                                    "end": entity_end,
                                    "text": entity_text,
                                    "label": entity_label
                                })
                                
                        except (ValueError, KeyError, TypeError) as e:
                            log_error(f"Error processing entity in annotation {record_id}: {e}")
                            continue
        
        # Create standardized record
        record = {
            "id": record_id,
            "text": text,
            "tokens": tokens,
            "labels": labels,
            "entities": entities,
            "ner_tags": [label for label in labels]  # Duplicate for compatibility
        }
        
        return record
        
    except Exception as e:
        log_error(f"Error creating record for annotation {record_id}: {e}")
        return None

def generate_huggingface_datasets(annotations: List[Dict]) -> bool:
    """Generate HuggingFace compatible dataset files in JSONL format only"""
    
    log_info(f"🔄 Converting {len(annotations)} annotations to HuggingFace format")
    
    dataset_records = []
    
    for idx, annotation in enumerate(annotations):
        record = create_standardized_record(annotation, idx)
        if record:
            dataset_records.append(record)
        else:
            log_error(f"Skipped invalid annotation at index {idx}")
    
    if not dataset_records:
        log_error("❌ No valid records generated for HuggingFace dataset")
        return False
    
    log_info(f"✅ Created {len(dataset_records)} valid dataset records")
    
    # Create train/test split
    train_size = max(1, int(0.8 * len(dataset_records)))  # Ensure at least 1 record in train
    train_data = dataset_records[:train_size]
    test_data = dataset_records[train_size:] if len(dataset_records) > 1 else dataset_records[:1]  # Fallback for small datasets
    
    # Save all files as JSONL only (HuggingFace preferred format)
    files_written = 0
    
    # Full dataset
    if safe_jsonl_write("huggingface_dataset/dataset.jsonl", dataset_records):
        files_written += 1
    
    # Train split
    if safe_jsonl_write("huggingface_dataset/train.jsonl", train_data):
        files_written += 1
    
    # Test split
    if safe_jsonl_write("huggingface_dataset/test.jsonl", test_data):
        files_written += 1
    
    # Original Label Studio format (keep as single JSON for reference)
    if safe_json_write("huggingface_dataset/label_studio_annotations.json", annotations):
        files_written += 1
    
    # Create dataset configuration file for HuggingFace
    config = {
        "dataset_name": "swahili-ner-dataset",
        "task": "token-classification",
        "language": "sw",
        "tags": ["named-entity-recognition", "swahili", "ner", "token-classification"],
        "features": {
            "id": "int32",
            "text": "string", 
            "tokens": {"dtype": "string", "sequence": True},
            "labels": {"dtype": "string", "sequence": True},
            "ner_tags": {"dtype": "string", "sequence": True},
            "entities": {"dtype": "string", "sequence": True}
        },
        "splits": {
            "train": len(train_data),
            "test": len(test_data)
        }
    }
    
    if safe_json_write("huggingface_dataset/dataset_info.json", config):
        files_written += 1
    
    log_info(f"📊 Dataset Statistics:")
    log_info(f"   • Total records: {len(dataset_records):,}")
    log_info(f"   • Train records: {len(train_data):,}")
    log_info(f"   • Test records: {len(test_data):,}")
    log_info(f"   • Files created: {files_written}")
    
    # Validate the JSONL files by attempting to read them back
    validation_passed = True
    for file_name in ["dataset.jsonl", "train.jsonl", "test.jsonl"]:
        file_path = f"huggingface_dataset/{file_name}"
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    if line.strip():
                        json.loads(line)  # Validate JSON
                        line_count += 1
                log_info(f"✅ Validated {file_name}: {line_count} valid records")
        except Exception as e:
            log_error(f"❌ Validation failed for {file_name}: {e}")
            validation_passed = False
    
    return validation_passed

def send_telegram_message(message, is_success=False):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        log_info("⚠️ Telegram credentials missing - skipping notification")
        return False
    
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage", 
            data={
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
        )
        if response.status_code == 200:
            emoji = "✅" if is_success else "📱"
            log_info(f"{emoji} Telegram notification sent successfully")
            return True
        else:
            log_error(f"Telegram API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        log_error(f"Failed to send Telegram notification: {e}")
        return False

def notify_failure(error_message):
    message = f"🚨 <b>February AI Pipeline FAILED</b>\n\n❌ Error: {error_message}\n\n🕐 Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
    send_telegram_message(message, is_success=False)

def notify_success(stats):
    message = f"""🎉 <b>February AI Pipeline SUCCESS!</b>

✅ <b>Pipeline completed successfully</b>

📊 <b>Stats:</b>
• Files processed: {stats.get('files_processed', 'N/A')}
• Files skipped: {stats.get('files_skipped', 'N/A')}
• Texts processed: {stats.get('texts_processed', 'N/A')}
• Annotations created: {stats.get('annotations_created', 'N/A')}
• Processing time: {stats.get('total_time', 'N/A')}s
• HuggingFace upload: {'✅ Success' if stats.get('hf_success') else '❌ Failed'}
• Label Studio import: {'✅ Success' if stats.get('ls_success') else '⚠️ Skipped'}

🔍 <b>Check Hugging Face</b> for your updated dataset!

🕐 <b>Completed:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"""
    send_telegram_message(message, is_success=True)

def generate_readme_with_metadata(dataset_name, num_samples, language, model_used, files_processed, texts_processed, processing_time, limits_applied):
    """Generate README.md file with dataset metadata"""
    
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    limits_section = ""
    if limits_applied:
        limits_section = f"""
## Processing Limits Applied

This dataset was processed with the following limits for performance optimization:
{chr(10).join([f"- **{k}**: {v}" for k, v in limits_applied.items()])}
"""
    
    readme_content = f"""# {dataset_name}

## Dataset Card

**Dataset Name:** {dataset_name}  
**Language:** {language} (Swahili)  
**Number of Samples:** {num_samples:,}  
**Model Used for Annotation:** {model_used}  
**Files Processed:** {files_processed}  
**Texts Processed:** {texts_processed:,}  
**Processing Time:** {processing_time:.2f} seconds  
**Generated:** {current_time} UTC  

## Description

This is an automatically annotated dataset for Swahili Named Entity Recognition (NER). The dataset was processed using the February AI Pipeline, which recursively discovers and processes JSON, JSONL, and TXT files from the data directory with performance optimizations for large datasets.

**All dataset files are in JSONL format** for maximum HuggingFace compatibility.
{limits_section}
## Dataset Structure

The dataset contains annotations in standardized JSONL format with the following entity types:
- **PERSON** - Names of people
- **ORGANIZATION** - Companies, institutions, organizations
- **LOCATION** - Places, cities, countries, geographical locations
- **MISCELLANEOUS** - Other named entities

## Processing Pipeline

1. **File Discovery**: Recursively scans `data/` directory for `.json`, `.jsonl`, and `.txt` files
2. **Smart Limiting**: Applies configurable limits for files, texts, and processing time
3. **Batch Processing**: Processes texts in batches to manage memory efficiently
4. **Text Extraction**: Handles various formats with consistent JSONL output
5. **Auto-Annotation**: Uses `{model_used}` for entity recognition
6. **Format Standardization**: Converts all outputs to HuggingFace-compatible JSONL
7. **Validation**: Validates all JSONL files for format correctness

## Files

- `dataset.jsonl` - Full dataset in JSONL format (HuggingFace compatible)
- `train.jsonl` - Training split (80% of data)
- `test.jsonl` - Test split (20% of data)
- `dataset_info.json` - HuggingFace dataset configuration
- `label_studio_annotations.json` - Original Label Studio format (reference only)
- `README.md` - This documentation

## Dataset Format

Each record in the JSONL files contains:

```json
{{
  "id": 0,
  "text": "Rais wa Tanzania Samia Suluhu Hassan amekutana na wabunge.",
  "tokens": ["Rais", "wa", "Tanzania", "Samia", "Suluhu", "Hassan", "amekutana", "na", "wabunge."],
  "labels": ["O", "O", "B-LOCATION", "B-PERSON", "I-PERSON", "I-PERSON", "O", "O", "O"],
  "ner_tags": ["O", "O", "B-LOCATION", "B-PERSON", "I-PERSON", "I-PERSON", "O", "O", "O"],
  "entities": [
    {{
      "start": 11,
      "end": 19,
      "text": "Tanzania",
      "label": "LOCATION"
    }},
    {{
      "start": 20,
      "end": 38,
      "text": "Samia Suluhu Hassan", 
      "label": "PERSON"
    }}
  ]
}}
```

### Label Format
- **BIO Tagging**: B- (Beginning), I- (Inside), O (Outside)
- **Entity Types**: PERSON, ORGANIZATION, LOCATION, MISCELLANEOUS

## Usage with HuggingFace

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset('json', data_files={{
    'train': 'train.jsonl',
    'test': 'test.jsonl'
}})

# Or load from HuggingFace Hub (after upload)
# dataset = load_dataset('your-username/swahili-ner-dataset')

# Access the data
print(dataset['train'][0])
```

## Manual Usage

```python
import json

# Load JSONL file
dataset = []
with open('dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        dataset.append(json.loads(line))

# Process records
for record in dataset:
    print(f"Text: {{record['text']}}")
    print(f"Tokens: {{record['tokens']}}")  
    print(f"Labels: {{record['labels']}}")
    print("---")
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@dataset{{swahili_ner_dataset_{current_time.replace('-', '').replace(':', '').replace(' ', '_')},
  title={{{dataset_name}}},
  author={{February AI Pipeline}},
  year={{{datetime.now().year}}},
  language={{{language}}},
  samples={{{num_samples}}},
  annotation_model={{{model_used}}},
  processing_time={{{processing_time:.2f}}},
  format={{JSONL}}
}}
```

## License

This dataset is provided as-is for research and educational purposes.

---

*Generated automatically by February AI Pipeline on {current_time} UTC*  
*Processing completed in {processing_time:.2f} seconds*  
*All files in JSONL format for HuggingFace compatibility*
"""

    # Save to huggingface_dataset directory
    readme_path = "huggingface_dataset/README.md"
    try:
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        log_info(f"✅ README.md generated at {readme_path}")
    except Exception as e:
        log_error(f"❌ Failed to generate README.md: {e}")

def import_to_label_studio():
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    url = os.getenv("LABEL_STUDIO_URL")
    project_id = os.getenv("PROJECT_ID")

    if not api_key or not url or not project_id:
        log_info("⚠️ Missing Label Studio credentials - skipping import")
        return False

    try:
        project_id = int(project_id)
        log_info(f"🔗 Connecting to Label Studio Project {project_id}")
        ls = Client(url=url, api_key=api_key)
        project = ls.get_project(project_id)
        project.import_tasks("annotations/all_swahili_annotations.json")
        log_info("✅ Successfully imported to Label Studio")
        return True
    except Exception as e:
        log_error(f"❌ Label Studio import failed: {e}")
        return False

def run(clear_skip=False, max_files=DEFAULT_MAX_FILES, max_texts_per_file=DEFAULT_MAX_TEXTS_PER_FILE, 
        max_total_texts=DEFAULT_MAX_TOTAL_TEXTS, max_text_length=DEFAULT_MAX_TEXT_LENGTH,
        max_file_size_mb=None, batch_size=20):
    """
    Main pipeline function with performance limits and JSONL standardization
    
    Args:
        clear_skip (bool): If True, clear the skip log and reprocess all files
        max_files (int): Maximum number of files to process
        max_texts_per_file (int): Maximum texts to extract per file
        max_total_texts (int): Maximum total texts to process across all files
        max_text_length (int): Maximum length of each text (characters)
        max_file_size_mb (float): Maximum file size in MB to process
        batch_size (int): Number of texts to process in each batch
    """
    start_time = time.time()
    
    stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'texts_processed': 0,
        'annotations_created': 0,
        'hf_success': False,
        'ls_success': False,
        'total_time': 0
    }

    limits_applied = {
        "Max Files": max_files,
        "Max Texts per File": max_texts_per_file,
        "Max Total Texts": max_total_texts,
        "Max Text Length": f"{max_text_length} characters",
        "Batch Size": batch_size,
        "Output Format": "JSONL only (HuggingFace optimized)"
    }
    
    if max_file_size_mb:
        limits_applied["Max File Size"] = f"{max_file_size_mb}MB"

    try:
        log_info("🚀 Starting February AI pipeline with JSONL standardization")
        log_info("⚡ Performance Limits:")
        for key, value in limits_applied.items():
            log_info(f"   • {key}: {value}")
        
        # Clear skip log if requested
        if clear_skip:
            clear_skip_log()
        
        # Create necessary directories
        os.makedirs("annotations", exist_ok=True)
        os.makedirs("huggingface_dataset", exist_ok=True)

        # Load skip log
        processed_files = load_skip_log()
        log_info(f"📋 Loaded skip log with {len(processed_files)} previously processed files")

        all_annotations = []
        all_texts = []
        total_texts_processed = 0

        # Find all dataset files recursively with limits
        dataset_files = find_dataset_files(max_files=max_files, max_file_size_mb=max_file_size_mb)
        
        if not dataset_files:
            log_error("❌ No dataset files found in data/ directory")
            return

        # Process each file
        for file_path in dataset_files:
            # Check if file was already processed
            file_hash = get_file_hash(file_path)
            if file_hash and file_hash in processed_files:
                log_info(f"⏭️ Skipping already processed file: {file_path}")
                stats['files_skipped'] += 1
                continue

            # Check if we've reached the total text limit
            if total_texts_processed >= max_total_texts:
                log_info(f"🔢 Reached maximum total texts limit ({max_total_texts})")
                break

            log_info(f"📂 Processing file: {file_path} ({get_file_size_mb(file_path):.1f}MB)")
            
            # Calculate remaining text quota
            remaining_quota = max_total_texts - total_texts_processed
            effective_max_per_file = min(max_texts_per_file, remaining_quota)
            
            # Load texts from file with limits
            texts = load_dataset_file(file_path, 
                                    max_texts=effective_max_per_file, 
                                    max_text_length=max_text_length)
            
            if not texts:
                log_info(f"⚠️ No texts found in {file_path}")
                continue
                
            stats['texts_processed'] += len(texts)
            total_texts_processed += len(texts)
            all_texts.extend(texts)

            log_info(f"🤖 Annotating {len(texts)} texts from {os.path.basename(file_path)}")
            
            # Process in batches for this file
            file_annotations = process_in_batches(texts, batch_size)
            all_annotations.extend(file_annotations)
            
            # Mark file as processed
            if file_hash:
                update_skip_log(file_hash)
            
            stats['files_processed'] += 1
            
            # Progress update
            elapsed = time.time() - start_time
            log_info(f"📊 Progress: {total_texts_processed}/{max_total_texts} texts, {elapsed:.1f}s elapsed")

        stats['annotations_created'] = len(all_annotations)
        stats['total_time'] = time.time() - start_time
        
        if not all_annotations:
            log_info("⚠️ No annotations created - nothing to save")
            return

        # Save combined annotations (Label Studio format for reference)
        export_path = "annotations/all_swahili_annotations.json"
        if safe_json_write(export_path, all_annotations):
            log_info(f"✅ Label Studio annotations saved to {export_path}")

        # Generate HuggingFace compatible datasets (JSONL only)
        log_info("🔄 Converting to HuggingFace JSONL format...")
        if generate_huggingface_datasets(all_annotations):
            log_info("✅ HuggingFace JSONL datasets generated successfully")
            stats['hf_format_success'] = True
        else:
            log_error("❌ Failed to generate HuggingFace datasets")
            stats['hf_format_success'] = False

        # Generate README.md with metadata
        generate_readme_with_metadata(
            dataset_name="swahili-ner-dataset",
            num_samples=stats['annotations_created'],
            language="sw",
            model_used="dslim/bert-base-NER",
            files_processed=stats['files_processed'],
            texts_processed=stats['texts_processed'],
            processing_time=stats['total_time'],
            limits_applied=limits_applied
        )

        # Push to Hugging Face
        try:
            log_info("🚀 Pushing to HuggingFace...")
            push_files()
            stats['hf_success'] = True
            log_info("✅ Hugging Face push complete")
        except Exception as e:
            log_error(f"❌ Hugging Face push failed: {e}")

        # Import to Label Studio
        stats['ls_success'] = import_to_label_studio()

        log_info("🎉 Pipeline completed successfully!")
        log_info(f"⏱️ Total processing time: {stats['total_time']:.2f} seconds")
        log_info("📁 All files generated in JSONL format for HuggingFace compatibility")
        log_info("🔍 **Check Hugging Face** for your updated dataset!")
        notify_success(stats)

    except Exception as e:
        error_msg = f"Pipeline failed: {str(e)}"
        log_error(f"❌ {error_msg}")
        notify_failure(error_msg)
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="February AI Pipeline - JSONL Optimized")
    parser.add_argument("--clear-skip", action="store_true", 
                       help="Clear the skip log and reprocess all files")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES,
                       help=f"Maximum number of files to process (default: {DEFAULT_MAX_FILES})")
    parser.add_argument("--max-texts-per-file", type=int, default=DEFAULT_MAX_TEXTS_PER_FILE,
                       help=f"Maximum texts per file (default: {DEFAULT_MAX_TEXTS_PER_FILE})")
    parser.add_argument("--max-total-texts", type=int, default=DEFAULT_MAX_TOTAL_TEXTS,
                       help=f"Maximum total texts to process (default: {DEFAULT_MAX_TOTAL_TEXTS})")
    parser.add_argument("--max-text-length", type=int, default=DEFAULT_MAX_TEXT_LENGTH,
                       help=f"Maximum text length in characters (default: {DEFAULT_MAX_TEXT_LENGTH})")
    parser.add_argument("--max-file-size", type=float, 
                       help="Maximum file size in MB to process")
    parser.add_argument("--batch-size", type=int, default=20,
                       help="Batch size for processing texts (default: 20)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: 5 files, 20 texts each, 100 total")
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.max_files = 5
        args.max_texts_per_file = 20
        args.max_total_texts = 100
        args.batch_size = 10
        print("🚀 Quick mode activated!")
    
    run(clear_skip=args.clear_skip,
        max_files=args.max_files,
        max_texts_per_file=args.max_texts_per_file,
        max_total_texts=args.max_total_texts,
        max_text_length=args.max_text_length,
        max_file_size_mb=args.max_file_size,
        batch_size=args.batch_size)