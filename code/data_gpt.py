import torch
import os
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)
from datasets import Dataset

# Set paths
BASE_DIR = os.path.expanduser("~/gpt2_experiment")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_books.txt")
SAVE_DIR = os.path.join(BASE_DIR, "custom_tokenizer")

# Ensure output directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "gpt2_checkpoints"), exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- STEP 1: TRAIN CUSTOM TOKENIZER --- #
tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[DATA_PATH],
    vocab_size=50257,
    min_frequency=3,
    special_tokens=["<|endoftext|>"]
)
tokenizer.save_model(SAVE_DIR, "custom_vocab")

# --- STEP 2: HUGGING FACE TOKENIZER --- #
hf_tokenizer = GPT2TokenizerFast(
    vocab_file=os.path.join(SAVE_DIR, "custom_vocab-vocab.json"),
    merges_file=os.path.join(SAVE_DIR, "custom_vocab-merges.txt")
)
hf_tokenizer.add_special_tokens({
    "eos_token": "<|endoftext|>",
    "bos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>"
})
hf_tokenizer.save_pretrained(SAVE_DIR)

hf_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, local_files_only=True)

# --- STEP 3: DATASET PREPARATION (TOKENIZE FIRST, THEN SPLIT, THEN CHUNK) --- #

# Read raw text
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split into books
books = raw_text.split("### BOOK SEPARATOR ###")

# Tokenize all books first
tokenized_books = [hf_tokenizer.encode(book.strip()) + [hf_tokenizer.eos_token_id] for book in books]

# Flatten all tokens into a single list
all_tokens = [token for book in tokenized_books for token in book]

# Define split sizes
total_tokens = len(all_tokens)
train_size = int(0.8 * total_tokens)
valid_size = int(0.1 * total_tokens)
test_size = total_tokens - train_size - valid_size  # Ensure 100%

# Assign tokenized chunks to respective splits
train_tokens = all_tokens[:train_size]
valid_tokens = all_tokens[train_size:train_size + valid_size]
test_tokens = all_tokens[train_size + valid_size:]

# Function to apply sliding window chunking
def chunk_tokens(tokens, chunk_size=1024, overlap=512):
    return [tokens[i:i + chunk_size] 
            for i in range(0, len(tokens) - chunk_size + 1, chunk_size - overlap)]

# Chunk each dataset separately
final_datasets = {
    "train": Dataset.from_dict({"input_ids": chunk_tokens(train_tokens)}),
    "validation": Dataset.from_dict({"input_ids": chunk_tokens(valid_tokens)}),
    "test": Dataset.from_dict({"input_ids": chunk_tokens(test_tokens)}),
}

# Convert to PyTorch format
for split in final_datasets:
    final_datasets[split].set_format(type="torch", columns=["input_ids"])

# Print dataset sizes
print(f"Train chunks: {len(final_datasets['train'])}")
print(f"Validation chunks: {len(final_datasets['validation'])}")
print(f"Test chunks: {len(final_datasets['test'])}")

# Print the first few examples
print("First few examples:")
for i in range(5):
    print(final_datasets["train"][i])

# Print the first few examples in decoded form
print("First few examples in decoded form:")
for i in range(5):
    print(hf_tokenizer.decode(final_datasets["train"][i]["input_ids"], skip_special_tokens=True))