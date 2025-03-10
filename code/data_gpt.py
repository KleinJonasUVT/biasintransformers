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
DATA_PATH = os.path.join(BASE_DIR, "cleaned_books_small.txt")
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
tokenizer.save_model(SAVE_DIR, "custom_vocab_small")

# --- STEP 2: HUGGING FACE TOKENIZER --- #
hf_tokenizer = GPT2TokenizerFast(
    vocab_file=os.path.join(SAVE_DIR, "custom_vocab_small-vocab.json"),
    merges_file=os.path.join(SAVE_DIR, "custom_vocab_small-merges.txt")
)
hf_tokenizer.add_special_tokens({
    "eos_token": "<|endoftext|>",
    "bos_token": "<|endoftext|>",
    "pad_token": "<|endoftext|>"
})
hf_tokenizer.save_pretrained(SAVE_DIR)

hf_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, local_files_only=True)

# --- STEP 3: DATASET PREPARATION --- #
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

books = raw_text.split("### BOOK SEPARATOR ###")

def tokenize_book(text, tokenizer):
    tokens = tokenizer.encode(text) + [tokenizer.eos_token_id]
    return tokens

def sliding_window(tokens, chunk_size=1024, overlap=512):
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens)-chunk_size+1, chunk_size-overlap)]

all_chunks = []
for book_text in books:
    book_tokens = tokenize_book(book_text.strip(), hf_tokenizer)
    all_chunks.extend(sliding_window(book_tokens))

print("Total chunks created:", len(all_chunks))

dataset = Dataset.from_dict({"input_ids": all_chunks})

# Split dataset
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
valid_test_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

final_datasets = {
    "train": train_test_split["train"],
    "validation": valid_test_split["train"],
    "test": valid_test_split["test"],
}

for split in final_datasets:
    final_datasets[split].set_format(type="torch", columns=["input_ids"])

print(f'Train size: {len(final_datasets["train"])}')

# Print the first few examples
print("First few examples:")
for i in range(5):
    print(final_datasets["train"][i])

# --- STEP 4: MODEL SETUP --- #
config = GPT2Config(
    vocab_size=len(hf_tokenizer),
    n_embd=768,
    n_layer=12,
    n_head=12,
    n_positions=1024,
    pad_token_id=hf_tokenizer.pad_token_id,
)

model = GPT2LMHeadModel(config)
model.resize_token_embeddings(len(hf_tokenizer))
model.to(device)
print(model)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer,
    mlm=False
)

# --- STEP 5: TRAINING ARGUMENTS & TRAINING --- #
training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "gpt2_checkpoints"),
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=10000,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=100,
    num_train_epochs=10,
    save_total_limit=None,
    overwrite_output_dir=False,
    logging_dir=os.path.join(BASE_DIR, "logs"),
    load_best_model_at_end=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["validation"],
    tokenizer=hf_tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(os.path.join(BASE_DIR, "gpt2_custom_final"))
print("Training complete, model saved to:", os.path.join(BASE_DIR, "gpt2_custom_final"))
