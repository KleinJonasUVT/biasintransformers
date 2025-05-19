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
nltk.download('punkt_tab')

# Set paths
BASE_DIR = os.path.expanduser("~/bert_experiment")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_books.txt")
SAVE_DIR = os.path.join(BASE_DIR, "custom_tokenizer")
VOCAB_FILE = os.path.join(SAVE_DIR, "custom_vocab-vocab.txt")

# Ensure output directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "bert_checkpoints"), exist_ok=True)

# Ensure GPU is used if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

### --- STEP 1: TRAIN CUSTOM TOKENIZER --- ###
# Initialize WordPiece tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False
)

# Train tokenizer
tokenizer.train(
    files=[DATA_PATH],
    vocab_size=30000,  # Large enough to cover most words, preventing `[UNK]`
    min_frequency=3,    # Avoid too many rare words being excluded
    limit_alphabet=1000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Save tokenizer in original format
tokenizer.save_model(SAVE_DIR, "custom_vocab")
print("WordPiece tokenizer training complete! Saved to", SAVE_DIR)

### --- STEP 2: CONVERT TO HUGGING FACE TOKENIZER FORMAT --- ###
hf_tokenizer = BertTokenizerFast(
    vocab_file=VOCAB_FILE,
    do_lower_case=False
)

# Add special tokens
hf_tokenizer.add_special_tokens({
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]"
})

# Save tokenizer in Hugging Face format
hf_tokenizer.save_pretrained(SAVE_DIR)
print("Tokenizer successfully saved in Hugging Face format at:", SAVE_DIR)

# Verify that necessary files exist
if not os.path.exists(os.path.join(SAVE_DIR, "tokenizer.json")):
    raise FileNotFoundError("Tokenizer files are missing! Ensure tokenizer.json and related files are present.")

### --- STEP 3: LOAD TOKENIZER SAFELY --- ###
hf_tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
print("Tokenizer loaded successfully!")

### --- STEP 4: READ DATA AND PREVENT BOOK MIXING --- ###
# Read books
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Define book delimiter
book_delimiter = "\n### BOOK SEPARATOR ###\n"
books = text.split(book_delimiter)

print("Number of books:", len(books))

# Shuffle books for randomness
random.seed(42)
random.shuffle(books)


### --- STEP 1: Tokenize books & Measure Token Counts --- ###
def tokenize_books(book_list, tokenizer):
    """Tokenizes books while respecting sentence boundaries."""
    tokenized_books = []
    token_counts = []
    
    for book in book_list:
        sentences = sent_tokenize(book)  # Split into sentences
        tokenized_sentences = [tokenizer.encode(sent, add_special_tokens=False) for sent in sentences]

        # Flatten list while keeping sentence boundaries
        tokenized_book = []
        for sent_tokens in tokenized_sentences:
            tokenized_book.extend(sent_tokens)
        
        tokenized_books.append(tokenized_sentences)  # List of lists
        token_counts.append(len(tokenized_book))  # Total tokens
    
    return tokenized_books, token_counts

tokenized_books, token_counts = tokenize_books(books, hf_tokenizer)

# Compute total tokens
total_tokens = sum(token_counts)
train_token_target = int(total_tokens * 0.80)
val_token_target = int(total_tokens * 0.10)
test_token_target = int(total_tokens * 0.10)

print(f"Total tokens: {total_tokens}, Train: {train_token_target}, Val: {val_token_target}, Test: {test_token_target}")


### --- STEP 2: Split Books While Respecting Token Distribution --- ###
def split_books_by_token_count(tokenized_books, token_counts, train_target, val_target, test_target):
    """Splits books while ensuring token-based distribution."""
    
    train_books, val_books, test_books = [], [], []
    train_count, val_count, test_count = 0, 0, 0
    
    for book_sentences, count in zip(tokenized_books, token_counts):
        if train_count + count <= train_target:
            train_books.append(book_sentences)
            train_count += count
        elif val_count + count <= val_target:
            val_books.append(book_sentences)
            val_count += count
        else:
            test_books.append(book_sentences)
            test_count += count
    
    return train_books, val_books, test_books

train_tokenized, val_tokenized, test_tokenized = split_books_by_token_count(
    tokenized_books, token_counts, train_token_target, val_token_target, test_token_target
)

# Verify distributions
print(f"Final token counts - Train: {sum(len(sum(b, [])) for b in train_tokenized)}, Val: {sum(len(sum(b, [])) for b in val_tokenized)}, Test: {sum(len(sum(b, [])) for b in test_tokenized)}")


### --- STEP 3: Chunk While Respecting Sentence Boundaries --- ###
def find_closest_sentence_start(sentences, tokenizer, target_token_count):
    """Find the closest sentence index that gets us to the target token count."""
    current_token_count = 0
    for i, sentence_tokens in enumerate(sentences):
        current_token_count += len(sentence_tokens)
        if current_token_count >= target_token_count:
            return i
    return len(sentences)  # Default to end

def chunk_sentences_respecting_boundaries(tokenized_sentences, tokenizer, max_length=512, stride=384):
    """Chunk text while ensuring sentence boundaries are respected."""
    chunks = []
    start = 0

    while start < len(tokenized_sentences):
        current_chunk = []
        current_length = 2  # [CLS] and [SEP]
        
        for i in range(start, len(tokenized_sentences)):
            sentence_tokens = tokenized_sentences[i]
            sentence_length = len(sentence_tokens)

            if current_length + sentence_length > max_length:
                break  # Stop at the last full sentence before max_length
            
            current_chunk.append(sentence_tokens)
            current_length += sentence_length
        
        # Flatten and add special tokens
        chunk = [tokenizer.cls_token_id] + sum(current_chunk, []) + [tokenizer.sep_token_id]
        chunks.append(chunk)

        # Move start forward, ensuring overlap
        start += i - start if i - start > 0 else 1  # Move at least 1 sentence forward

    return chunks

# Apply chunking separately for train, val, and test
train_chunks = sum([chunk_sentences_respecting_boundaries(book, hf_tokenizer) for book in train_tokenized], [])
val_chunks = sum([chunk_sentences_respecting_boundaries(book, hf_tokenizer) for book in val_tokenized], [])
test_chunks = sum([chunk_sentences_respecting_boundaries(book, hf_tokenizer) for book in test_tokenized], [])

print(f"Total train chunks: {len(train_chunks)}, Val chunks: {len(val_chunks)}, Test chunks: {len(test_chunks)}")


### --- STEP 4: Convert to Dataset Format --- ###
def create_dataset(chunks, tokenizer):
    """Converts chunked data into a Dataset object with padding."""

    # Pad sequences
    padded_chunks = pad_sequence(
        [torch.tensor(chunk) for chunk in chunks],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    return Dataset.from_dict({"input_ids": padded_chunks.tolist()})

train_dataset = create_dataset(train_chunks, hf_tokenizer)
val_dataset = create_dataset(val_chunks, hf_tokenizer)
test_dataset = create_dataset(test_chunks, hf_tokenizer)

# Store datasets in a dictionary
final_datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})

# Save datasets to disk
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "train_dataset")
VALID_DATA_PATH = os.path.join(BASE_DIR, "valid_dataset")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_dataset")

final_datasets["train"].save_to_disk(TRAIN_DATA_PATH)
final_datasets["validation"].save_to_disk(VALID_DATA_PATH)
final_datasets["test"].save_to_disk(TEST_DATA_PATH)

print("Datasets saved successfully!")

# Convert datasets to PyTorch format
for split in final_datasets:
    final_datasets[split].set_format(type="torch", columns=["input_ids"])

# Define a new BERT model configuration
config = BertConfig(
    vocab_size=len(hf_tokenizer),
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    type_vocab_size=1,
    pad_token_id=hf_tokenizer.pad_token_id,
    bos_token_id=hf_tokenizer.cls_token_id,
    eos_token_id=hf_tokenizer.sep_token_id
)

# Initialize BERT model
model = BertForMaskedLM(config)
model.resize_token_embeddings(len(hf_tokenizer))

# Move model to GPU
model.to(device)
print(model)

# Define MLM data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer,
    mlm=True,
    mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "bert_checkpoints"),
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=296,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=296,
    num_train_epochs=200,
    save_total_limit=None,
    overwrite_output_dir=False,
    logging_dir=os.path.join(BASE_DIR, "logs"),
    load_best_model_at_end=True,
    fp16=True,
    fp16_full_eval=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["validation"],
    processing_class=hf_tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model(os.path.join(BASE_DIR, "bert_custom_final"))