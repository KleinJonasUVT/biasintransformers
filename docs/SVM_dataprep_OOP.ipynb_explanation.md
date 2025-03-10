# Explanation in a minimal flowchart of what happens in the SVM_dataprep_OOP.ipynb file

```plaintext
Start
  │
  ▼
Initialize `WordEmbeddingExtractor`
  ├─ Load Tokenizer & Model
  ├─ Load Dataset from File
  ▼
User Calls `process_multiple_words`
  ├─ For Each (word, label) in List:
  │    ├─ Call `extract_embeddings(word, label)`
  │    │    ├─ Call `filter_dataset(word)`
  │    │    │    ├─ Get Token IDs of the Word
  │    │    │    ├─ Filter Dataset Rows Containing the Word
  │    │    │    └─ Return Filtered DataFrame & Word IDs
  │    │    ├─ For Each Row in Filtered DataFrame:
  │    │    │    ├─ Find Word's Position in Input IDs
  │    │    │    ├─ Convert Input IDs to Text
  │    │    │    ├─ Tokenize Text for Model Input
  │    │    │    ├─ Pass Tokens Through Model to Get Hidden States
  │    │    │    ├─ Extract Word Embeddings from Last Hidden State
  │    │    │    ├─ Store Embeddings
  │    │    └─ Return Embeddings DataFrame
  │    ├─ Store Embeddings and Labels in Lists
  ▼
Save Embeddings and Labels as .npy Files
  ▼
End
```
