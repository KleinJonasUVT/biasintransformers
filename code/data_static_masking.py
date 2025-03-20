import torch
import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
import nltk
from nltk.tokenize import sent_tokenize
import random

nltk.download("punkt")

# Set paths
BASE_DIR = os.path.expanduser("~/bert_static_masking")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_books_small.txt")
SAVE_DIR = os.path.join(BASE_DIR, "custom_tokenizer")
VOCAB_FILE = os.path.join(SAVE_DIR, "custom_vocab_small-vocab.txt")

# Ensure output directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "bert_static_checkpoints"), exist_ok=True)

# Ensure GPU is used if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

### --- STEP 1: TRAIN CUSTOM TOKENIZER --- ###
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)

tokenizer.train(
    files=[DATA_PATH],
    vocab_size=30000,
    min_frequency=3,
    limit_alphabet=1000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.save_model(SAVE_DIR, "custom_vocab_small")
print("WordPiece tokenizer training complete! Saved to", SAVE_DIR)

hf_tokenizer = BertTokenizerFast(
    vocab_file=VOCAB_FILE,
    do_lower_case=False
)

hf_tokenizer.add_special_tokens({
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]"
})

hf_tokenizer.save_pretrained(SAVE_DIR)
print("Tokenizer successfully saved in Hugging Face format at:", SAVE_DIR)

hf_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
print("Tokenizer loaded successfully!")

# Read and preprocess data
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

book_delimiter = "\n### BOOK SEPARATOR ###\n"
books = text.split(book_delimiter)
random.seed(42)
random.shuffle(books)

### --- Tokenization --- ###
def tokenize_books(book_list, tokenizer):
    tokenized_books = []
    token_counts = []
    
    for book in book_list:
        sentences = sent_tokenize(book)
        tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]
        tokenized_book = sum(tokenized_sentences, [])
        tokenized_books.append(tokenized_sentences)
        token_counts.append(len(tokenized_book))
    
    return tokenized_books, token_counts

tokenized_books, token_counts = tokenize_books(books, hf_tokenizer)

### --- Splitting and Chunking Data --- ###
train_books, val_books, test_books = [], [], []
train_count, val_count, test_count = 0, 0, 0

total_tokens = sum(token_counts)
train_target, val_target, test_target = int(0.8 * total_tokens), int(0.1 * total_tokens), int(0.1 * total_tokens)

for book, count in zip(tokenized_books, token_counts):
    if train_count + count <= train_target:
        train_books.append(book)
        train_count += count
    elif val_count + count <= val_target:
        val_books.append(book)
        val_count += count
    else:
        test_books.append(book)
        test_count += count

def chunk_books(book_list, tokenizer, max_length=512, stride=384):
    chunks = []
    
    for book in book_list:
        current_chunk = []
        current_length = 2  # Accounting for [CLS] and [SEP]
        
        for sentence in book:
            sentence_length = len(sentence)

            # If adding this sentence exceeds max_length, finalize the current chunk
            if current_length + sentence_length > max_length:
                chunk = [tokenizer.cls_token_id] + sum(current_chunk, []) + [tokenizer.sep_token_id]
                chunks.append(chunk)

                # Start a new chunk with proper stride
                overlap = current_chunk[-(stride // 2):]  # Take last half as stride
                current_chunk = overlap if overlap else []  # Ensure no empty lists
                current_length = sum(len(sent) for sent in current_chunk) + 2  # Reset length

            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add any remaining chunk at the end
        if current_chunk:
            chunk = [tokenizer.cls_token_id] + sum(current_chunk, []) + [tokenizer.sep_token_id]
            chunks.append(chunk)

    return chunks

train_chunks = chunk_books(train_books, hf_tokenizer)
val_chunks = chunk_books(val_books, hf_tokenizer)
test_chunks = chunk_books(test_books, hf_tokenizer)

print(len(train_chunks[0]))  # Should be ≤ 512
print(f"Chunk type: {type(train_chunks[0])}")

### --- Static Masking --- ###
def static_masking(inputs, tokenizer, mask_ratio=0.15):
    """Apply a fixed mask at the same positions across batches."""
    inputs = torch.tensor(inputs, dtype=torch.long)
    labels = inputs.clone()

    # Ensure batch dimension
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)  # Add batch dimension

    batch_size, seq_len = inputs.shape  # Get batch and sequence length

    # Create mask tensor with correct shape
    mask = torch.full((batch_size, seq_len), False, dtype=torch.bool)  # Ensure 2D mask

    for i in range(batch_size):  # Loop over batch
        if seq_len == 0:  # Skip empty sequences
            continue
        num_to_mask = max(1, int(mask_ratio * seq_len))  # Avoid zero masks
        mask_indices = torch.randperm(seq_len)[:num_to_mask]  # Random mask indices
        mask[i, mask_indices] = True  # Apply mask correctly

    inputs[mask] = tokenizer.mask_token_id

    # Ensure `labels` is also 2D before applying the mask
    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels  # Ensure 2D shape
    labels[~mask] = -100  # Ignore loss for non-masked tokens

    return {"input_ids": inputs.squeeze(0).tolist(), "labels": labels.squeeze(0).tolist()}

train_chunks_masked = [static_masking(chunk, hf_tokenizer) for chunk in train_chunks]
val_chunks_masked = [static_masking(chunk, hf_tokenizer) for chunk in val_chunks]
test_chunks_masked = [static_masking(chunk, hf_tokenizer) for chunk in test_chunks]

# Convert to datasets
train_dataset = Dataset.from_dict({"input_ids": [item["input_ids"] for item in train_chunks_masked], "labels": [item["labels"] for item in train_chunks_masked]})
val_dataset = Dataset.from_dict({"input_ids": [item["input_ids"] for item in val_chunks_masked], "labels": [item["labels"] for item in val_chunks_masked]})
test_dataset = Dataset.from_dict({"input_ids": [item["input_ids"] for item in test_chunks_masked], "labels": [item["labels"] for item in test_chunks_masked]})

final_datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Set format to PyTorch
final_datasets.set_format(type="torch", columns=["input_ids", "labels"])

# Print dataset sizes
print(f"Train chunks: {len(final_datasets['train'])}")
print(f"Validation chunks: {len(final_datasets['validation'])}")
print(f"Test chunks: {len(final_datasets['test'])}")