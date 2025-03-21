import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import os
from gensim.models.callbacks import CallbackAny2Vec

# Set paths
BASE_DIR = os.path.expanduser("~/word2vec")
DATA_PATH = os.path.join(BASE_DIR, "cleaned_books.txt")

# Ensure output directories exist
os.makedirs(BASE_DIR, exist_ok=True)

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

# Improved preprocessing: Tokenize, remove stopwords, lowercase, filter non-alpha tokens
tokenized_sentences = []
for book in books:
    sentences = sent_tokenize(book)
    for sent in sentences:
        words = [word.lower() for word in word_tokenize(sent) if word.isalpha() and word.lower() not in stop_words]
        if words:  # ensure sentence is not empty after filtering
            tokenized_sentences.append(words)

print(f"Total tokenized sentences after filtering: {len(tokenized_sentences)}")

class EpochLogger(CallbackAny2Vec):
    """Callback to log loss after each epoch."""
    def __init__(self):
        self.epoch = 0
        self.previous_loss = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        current_loss = loss - self.previous_loss if self.epoch > 0 else loss
        print(f"Loss after epoch {self.epoch}: {current_loss}")
        self.previous_loss = loss
        self.epoch += 1

# Initialize the loss logger
epoch_logger = EpochLogger()

# Train improved Word2Vec model
word2vec_model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=200,
    window=8,
    min_count=5,
    workers=4,
    sg=1,
    epochs=20,
    compute_loss=True, 
    callbacks=[epoch_logger]
)

# Save the model
WORD2VEC_SAVE_PATH = os.path.join(BASE_DIR, "word2vec_model")

os.makedirs(os.path.dirname(WORD2VEC_SAVE_PATH), exist_ok=True)

word2vec_model.save(WORD2VEC_SAVE_PATH)

print(f"Improved Word2Vec model trained and saved at: {WORD2VEC_SAVE_PATH}")
