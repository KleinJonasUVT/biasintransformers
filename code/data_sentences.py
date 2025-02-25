import torch
import os
from tokenizers import BertWordPieceTokenizer
from transformers import AutoTokenizer, BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import sent_tokenize

# Set paths
BASE_DIR = os.path.expanduser("~/bert_experiment")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_books_small.txt")
SAVE_DIR = os.path.join(BASE_DIR, "custom_tokenizer")
VOCAB_FILE = os.path.join(SAVE_DIR, "custom_vocab_small-vocab.txt")

# Ensure output directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "bert_checkpoints"), exist_ok=True)

# Ensure GPU is used if available (this is for on the aurometalsaurus server with a GPU, but irrelevant for this standalone script)
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
    vocab_size=30000,
    min_frequency=3,
    limit_alphabet=1000,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Save tokenizer in original format
tokenizer.save_model(SAVE_DIR, "custom_vocab_small")
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
# Read and split text into books
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# The text file consist of multiple books separated by the delimiter defined below, so splitting will work
book_delimiter = "\n### BOOK SEPARATOR ###\n"
books = text.split(book_delimiter)

# print a few examples
print("Number of books:", len(books))
print("Number of sentences in first book:", len(sent_tokenize(books[0])))
print("Number of sentences in second book:", len(sent_tokenize(books[1])))
print("Number of sentences in third book:", len(sent_tokenize(books[2])))

def find_closest_sentence_start(sentences, tokenizer, target_token_count):
    """Find the sentence index closest to the target token count"""
    current_token_count = 0
    for i, sentence in enumerate(sentences):
        tokenized_sentence = tokenizer.tokenize(sentence)
        current_token_count += len(tokenized_sentence)
        if current_token_count >= target_token_count:
            return i
    return len(sentences)

def chunk_sentences_with_overlap(text, tokenizer, max_length=512):
    """Split text into overlapping chunks of up to 512 tokens"""
    sentences = sent_tokenize(text, language="dutch")
    
    # Identify starting points for overlapping chunks
    start_indices = [
        0,
        find_closest_sentence_start(sentences, tokenizer, 128),
        find_closest_sentence_start(sentences, tokenizer, 256),
        find_closest_sentence_start(sentences, tokenizer, 384)
    ]
    
    all_chunks = []
    for start_index in start_indices:
        current_chunk, current_length = [], 2  # Account for [CLS] and [SEP], therefore already having 2 tokens
        
        for sentence in sentences[start_index:]:
            tokenized_sentence = tokenizer.tokenize(sentence)
            sentence_length = len(tokenized_sentence)

            if current_length + sentence_length > max_length:
                # Finalize the current chunk
                chunk_with_special_tokens = (
                    [tokenizer.cls_token] + current_chunk + [tokenizer.sep_token]
                )
                all_chunks.append(chunk_with_special_tokens)
                
                # Start a new chunk
                current_chunk, current_length = [], 2  # Reset with CLS and SEP tokens

            current_chunk.extend(tokenized_sentence)
            current_length += sentence_length

        if current_chunk:
            # Add special tokens to the last chunk
            chunk_with_special_tokens = (
                [tokenizer.cls_token] + current_chunk + [tokenizer.sep_token]
            )
            all_chunks.append(chunk_with_special_tokens)

    return all_chunks

def tokenize_and_pad(chunks, tokenizer, max_length=512):
    """ 
    Tokenize and pad a list of chunks using the huggingface tokenizer.
    """
    tokenized_chunks = [
        tokenizer.convert_tokens_to_ids(chunk) for chunk in chunks
    ]
    
    # Pad sequences
    padded_chunks = pad_sequence(
        [torch.tensor(chunk) for chunk in tokenized_chunks],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )

    return padded_chunks

# Process all books with overlapping chunks
all_books_chunks = []
for book in books:
    book_chunks = chunk_sentences_with_overlap(book, hf_tokenizer, max_length=512)
    padded_book_chunks = tokenize_and_pad(book_chunks, hf_tokenizer, max_length=512)
    all_books_chunks.append(padded_book_chunks)

print("Chunking and tokenization complete!")
print("Number of books processed:", len(all_books_chunks))

# Ensure all chunks have exactly 512 tokens
flattened_chunks = []
for book in all_books_chunks:
    for chunk in book:
        chunk = chunk[:512]  # Trim if it's too long
        if len(chunk) < 512:
            chunk += [0] * (512 - len(chunk))  # Pad if too short
        flattened_chunks.append(torch.tensor(chunk))

# Stack into a single tensor
all_books_chunks_tensor = torch.stack(flattened_chunks)

# Print the shape
print(all_books_chunks_tensor.shape)