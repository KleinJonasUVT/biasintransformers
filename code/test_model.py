import torch
import os
import time
from transformers import BertForMaskedLM, AutoTokenizer
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from azure.storage.blob import BlobServiceClient

# Azure connection setup
AZURE_CONNECTION_STRING = os.getenv("SONAR_STORAGE_KEY")
CONTAINER_NAME = "results"
BLOB_PREFIX = "bert_dynamic_full"

# Define local storage paths
BASE_LOCAL_PATH = "/Users/jonasklein/bert_evaluation"
CHECKPOINT_PATH = os.path.join(BASE_LOCAL_PATH, "bert_checkpoints", "checkpoint-592000")
TEST_DATA_PATH = os.path.join(BASE_LOCAL_PATH, "test_dataset")
TOKENIZER_PATH = os.path.join(BASE_LOCAL_PATH, "custom_tokenizer")

# Ensure directories exist
os.makedirs(BASE_LOCAL_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
os.makedirs(TOKENIZER_PATH, exist_ok=True)
os.makedirs(TEST_DATA_PATH, exist_ok=True)

# Function to download blobs from Azure with retries
def download_from_azure(blob_path, local_path, max_retries=3, chunk_size=4 * 1024 * 1024):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
    
    retries = 0
    while retries < max_retries:
        try:
            with open(local_path, "wb") as download_file:
                stream = blob_client.download_blob()
                total_size = stream.size
                downloaded = 0
                for chunk in stream.chunks():
                    download_file.write(chunk)
                    downloaded += len(chunk)
                    print(f"Downloading {blob_path}: {downloaded}/{total_size} bytes", end="\r")
                print(f"Downloaded {blob_path} to {local_path}")
            return
        except Exception as e:
            retries += 1
            print(f"Download failed ({retries}/{max_retries}): {e}")
            time.sleep(2 ** retries)
    raise RuntimeError(f"Failed to download {blob_path} after {max_retries} attempts.")

# Download model checkpoint files
checkpoint_files = [
    "config.json", "generation_config.json", "model.safetensors", "optimizer.pt",
    "rng_state.pth", "scheduler.pt", "special_tokens_map.json", "tokenizer_config.json",
    "tokenizer.json", "trainer_state.json", "training_args.bin", "vocab.txt"
]

for file in checkpoint_files:
    azure_path = f"{BLOB_PREFIX}/bert_checkpoints/checkpoint-592000/{file}"
    local_path = os.path.join(CHECKPOINT_PATH, file)
    download_from_azure(azure_path, local_path)

# Download tokenizer files
tokenizer_files = ["tokenizer.json", "vocab.txt", "tokenizer_config.json"]
for file in tokenizer_files:
    azure_path = f"{BLOB_PREFIX}/custom_tokenizer/{file}"
    local_path = os.path.join(TOKENIZER_PATH, file)
    download_from_azure(azure_path, local_path)

# Download test dataset files
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)
test_dataset_prefix = f"{BLOB_PREFIX}/test_dataset/"

blob_list = container_client.list_blobs(name_starts_with=test_dataset_prefix)
for blob in blob_list:
    blob_name = blob.name.replace(test_dataset_prefix, "")  # Get relative path
    local_file_path = os.path.join(TEST_DATA_PATH, blob_name)
    download_from_azure(blob.name, local_file_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
print("Tokenizer loaded successfully!")

# Load model
model = BertForMaskedLM.from_pretrained(CHECKPOINT_PATH)
print("Model loaded successfully!")

# Load test dataset
test_dataset = load_from_disk(TEST_DATA_PATH)

# Set dataset format for PyTorch
test_dataset.set_format(type="torch", columns=["input_ids"])

# Define MLM data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=BASE_LOCAL_PATH,
    per_device_eval_batch_size=8,
    dataloader_drop_last=False,
    report_to="none"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

# Evaluate model and compute loss
eval_results = trainer.evaluate()
print(f"Test Loss: {eval_results['eval_loss']}")
