import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient

# Set paths
BASE_DIR = os.path.expanduser("~/word2vec")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_books.txt")

# Azure setup
AZURE_STORAGE_CONNECTION_STRING = os.getenv("SONAR_STORAGE_KEY")
CONTAINER_NAME = "results"
BLOB_BASE_DIR = "w2v"

# Initialize blob client
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Ensure required NLTK models are downloaded
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

# Read raw text
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Split books using custom delimiter
book_delimiter = "\n### BOOK SEPARATOR ###\n"
books = text.split(book_delimiter)

# Preprocess text
tokenized_sentences = []
for book in books:
    sentences = sent_tokenize(book)
    for sent in sentences:
        words = [word.lower() for word in word_tokenize(sent) if word.isalpha() and word.lower() not in stop_words]
        if words:
            tokenized_sentences.append(words)

print(f"Total tokenized sentences after filtering: {len(tokenized_sentences)}")

# Word2Vec training parameters
vector_size = 200
window = 8
min_count = 5
workers = 4
epochs = 20
sub_epoch_fraction = 0.05  # 5%

# Reinitialize model
word2vec_model = Word2Vec(
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    workers=workers,
    sg=1,
)

# Build vocabulary
word2vec_model.build_vocab(tokenized_sentences)
print("Vocabulary built.")

# How many sub-updates per epoch (20 updates per epoch = 5% intervals)
num_updates = int(1 / sub_epoch_fraction)

# Start training in 5% chunks
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    for i in range(num_updates):
        start = int(i * len(tokenized_sentences) / num_updates)
        end = int((i + 1) * len(tokenized_sentences) / num_updates)
        partial_sentences = tokenized_sentences[start:end]

        word2vec_model.train(
            partial_sentences,
            total_examples=len(partial_sentences),
            epochs=1
        )

        # Save model to BytesIO buffer
        buffer = BytesIO()
        word2vec_model.save(buffer)
        buffer.seek(0)

        # Construct path: w2v/epoch{epoch+1}_part{i+1}/model
        subfolder = f"epoch{epoch+1}_part{i+1}"
        blob_path = f"{BLOB_BASE_DIR}/{subfolder}/model"

        # Upload buffer to Azure Blob
        blob_client = container_client.get_blob_client(blob_path)
        blob_client.upload_blob(buffer, overwrite=True)

        print(f"Checkpoint uploaded to Azure: {blob_path}")

# Save final model
final_buffer = BytesIO()
word2vec_model.save(final_buffer)
final_buffer.seek(0)
final_blob_path = f"{BLOB_BASE_DIR}/final/model"
final_blob_client = container_client.get_blob_client(final_blob_path)
final_blob_client.upload_blob(final_buffer, overwrite=True)

print(f"\nFinal Word2Vec model uploaded to Azure: {final_blob_path}")