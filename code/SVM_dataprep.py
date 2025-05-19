import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
import json
from azure.storage.blob import BlobServiceClient

class WordEmbeddingExtractor:
    def __init__(self, model_dir, dataset_path):
        print(f"Initializing WordEmbeddingExtractor with model_dir: {model_dir} and dataset_path: {dataset_path}")

        required_files = ["config.json", "tokenizer.json", "model.safetensors"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
        
        if missing_files:
            print(f"Error: Missing files in {model_dir}: {missing_files}")
            raise FileNotFoundError(f"Missing model files: {missing_files}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, model_max_length=512, truncation=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()

        print("Loading dataset...")
        self.dataset = Dataset.from_file(dataset_path)
        print(f"Loaded dataset with {len(self.dataset)} entries.")
    
    def get_word_ids(self, word):
        print(f"Getting token IDs for word: {word}")
        return self.tokenizer.encode(word, add_special_tokens=False)

    def filter_dataset(self, word):
        print(f"Filtering dataset for occurrences of word: {word}")
        word_ids = self.get_word_ids(word)
        
        filtered_examples = [
            example for example in self.dataset
            if any(
                example["input_ids"][i:i + len(word_ids)] == word_ids
                for i in range(len(example["input_ids"]) - len(word_ids) + 1)
            )
        ]
        print(f"Found {len(filtered_examples)} occurrences of '{word}' in the dataset.")
        return pd.DataFrame(filtered_examples), word_ids

    def extract_embeddings(self, word, word_label):
        print(f"Extracting embeddings for word: {word} with label: {word_label}")
        filtered_df, word_ids = self.filter_dataset(word)

        if filtered_df.empty:
            print(f"No occurrences of '{word}' found.")
            return None

        embeddings_dict = {}
        for index, row in filtered_df.iterrows():
            input_ids = row["input_ids"]
            word_indices = [
                i for i in range(len(input_ids) - len(word_ids) + 1)
                if input_ids[i:i+len(word_ids)] == word_ids
            ]
            text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            row_embeddings = []
            for idx in word_indices:
                word_span = list(range(idx, idx + len(word_ids)))
                word_embedding = last_hidden_state[:, word_span, :].mean(dim=1)
                row_embeddings.append(word_embedding.squeeze().numpy())
            embeddings_dict[index] = row_embeddings

        embeddings_df = pd.DataFrame(list(embeddings_dict.items()), columns=["row_index", "embeddings"])
        embeddings_df["label"] = word_label
        return embeddings_df

    def process_multiple_words(self, words_labels):
        print("Processing multiple words...")
        all_embeddings = []
        labels = []
        words_list = []
        for word, label in words_labels:
            print(f"Processing '{word}' with label {label}...")
            word_df = self.extract_embeddings(word, label)
            if word_df is not None:
                for _, row in word_df.iterrows():
                    for embedding in row["embeddings"]:
                        all_embeddings.append(embedding)
                        labels.append(label)
                        words_list.append(word)
        return np.array(all_embeddings), np.array(labels), words_list


# Azure Configuration
AZURE_CONNECTION_STRING = os.getenv("SONAR_STORAGE_KEY")
BLOB_CONTAINER_NAME = "results"
SVM_CONTAINER_NAME = "svmdata"
DATASET_PATH = "bert_dynamic_full/train_dataset"
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
svm_container_client = blob_service_client.get_container_client(SVM_CONTAINER_NAME)

# Download dataset files
local_dataset_dir = "temp_dataset"
os.makedirs(local_dataset_dir, exist_ok=True)
dataset_files = ["data-00000-of-00001.arrow", "dataset_info.json", "state.json"]
for file in dataset_files:
    blob_client = container_client.get_blob_client(f"{DATASET_PATH}/{file}")
    local_file_path = os.path.join(local_dataset_dir, file)
    with open(local_file_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

# Process checkpoints
checkpoints_prefix = "bert_dynamic_full/bert_checkpoints/"
checkpoints = list(set(blob.name.split('/')[2] for blob in container_client.list_blobs(name_starts_with=checkpoints_prefix)))

words_labels = [
    ("man", 1),
    ("vrouw", 0),
    ("broer", 1),
    ("zus", 0),
    ("zoon", 1),
    ("dochter", 0),
    ("neef", 1),
    ("nicht", 0),
    ("vader", 1),
    ("moeder", 0),
    ("opa", 1),
    ("oma", 0),
    ("kleinzoon", 1),
    ("kleindochter", 0),
    ("grootvader", 1),
    ("grootmoeder", 0),
    ("oom", 1),
    ("tante", 0),
    ("papa", 1),
    ("mama", 0),
    ("jongen", 1),
    ("meisje", 0),
    ("jongetje", 1),
    ("meid", 0),
    ("schoonvader", 1),
    ("schoonmoeder", 0),
    ("schoonzoon", 1),
    ("schoondochter", 0),
    ("stiefvader", 1),
    ("stiefmoeder", 0),
    ("stiefzoon", 1),
    ("stiefdochter", 0),
    ("peetvader", 1),
    ("peetmoeder", 0),
    ("bruidegom", 1),
    ("bruid", 0),
    ("meneer", 1),
    ("mevrouw", 0),
    ("mijnheer", 1),
    ("heer", 1),
    ("dame", 0),
    ("kerel", 1),
    ("mister", 1),
    ("miss", 0),
    ("mr", 1),
    ("ms", 0),
    ("prins", 1),
    ("prinses", 0),
    ("koning", 1),
    ("koningin", 0),
    ("lord", 1),
    ("lady", 0),
    ("baron", 1),
    ("barones", 0),
    ("hertog", 1),
    ("hertogin", 0),
    ("monnik", 1),
    ("non", 0),
    ("hijzelf", 1),
    ("zijzelf", 0),
    ("mannelijk", 1),
    ("vrouwelijk", 0),
    ("gentleman", 1),
    ("gozer", 1),
    ("wijf", 0)
]
for checkpoint in checkpoints:
    print(f"Processing checkpoint: {checkpoint}")
    local_model_dir = f"temp_model/{checkpoint}"
    os.makedirs(local_model_dir, exist_ok=True)
    
    model_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    for file in model_files:
        blob_path = f"bert_dynamic_full/bert_checkpoints/{checkpoint}/{file}"
        blob_client = container_client.get_blob_client(blob_path)
        local_file_path = os.path.join(local_model_dir, file)
        try:
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        except Exception as e:
            print(f"Error downloading {file} from Azure: {e}")

    downloaded_files = os.listdir(local_model_dir)
    if not all(f in downloaded_files for f in model_files):
        print(f"Error: Some model files are missing in {local_model_dir}.")
        continue

    extractor = WordEmbeddingExtractor(local_model_dir, os.path.join(local_dataset_dir, "data-00000-of-00001.arrow"))
    embeddings, labels, word_list = extractor.process_multiple_words(words_labels)

    if embeddings.size > 0 and labels.size > 0:
        embeddings_filename = f"embeddings_{checkpoint}.npy"
        labels_filename = f"labels_{checkpoint}.npy"

        np.save(embeddings_filename, embeddings)
        np.save(labels_filename, labels)

        # Upload to Azure
        for file in [embeddings_filename, labels_filename]:
            blob_client = svm_container_client.get_blob_client(file)
            with open(file, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            print(f"Uploaded {file} to Azure.")

        # Cleanup local files
        os.remove(embeddings_filename)
        os.remove(labels_filename)

print("Processing complete for all checkpoints.")