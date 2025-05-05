# Quantifying Gender Bias in Dutch Word Embeddings

This repository contains the code and analysis for my Data Science & Society thesis on detecting and quantifying gender bias in Dutch word embeddings. The project leverages a BERT Transformer model to track gender representation in embeddings, employing SVM-derived gender subspaces to analyze localization and evolution of biases over time. The research uses the SoNaR-corpus.

![Bert&Ernie](assets/images/Bert&Ernie.JPG)

## Data pipeline

![Data Pipeline](assets/images/flowchart.pdf)

## Contents
- **Data Preprocessing**: Scripts for preparing the SoNaR-corpus, including tokenization and cleaning.
- **Model Training**: Implementation of BERT Transformer model for creating Dutch word embeddings.
- **Bias Detection**: Classifiers and SVM to identify and quantify gender bias.
- **Analysis**: Analyzing the evolution and localization of gender bias in embeddings.
- **Evaluation**: Visualizations and results documenting embedding behaviors and gender localization.

## Repository structure

| File (in code folder)              | Description |
|-----------------------------------|-------------|
| `bert.ipynb`                      | First experimental code with BERT, not the final script |
| `corpus_to_azure.py`             | Script to upload parts of the local corpus to Azure |
| `data_exploration_lemma.ipynb`   | Data exploration at the lemma level (incomplete) |
| `data_exploration.ipynb`         | Exploratory Data Analysis (EDA) on the corpus |
| `data_sentences.py`              | Data preparation for my BERT Model, respecting both book and sentence boundaries |
| `visualize_results.ipynb`        | Visualization of model training results |
| `word2vec.py`                    | Training a word2vec model to get static embeddings from our corpus |
| `results_to_azure.py`            | Script that gets all model results to an Azure container for storage |
| `test_model.py`                  | Getting test accuracy for pretrained BERT model |
| `SVM_dataprep_OOP.ipynb`         | Processes multiple words to extract their contextual embeddings from BERT, and saves the results for later use in SVM training |
| `SVM_BERT_full.ipynb` | Training an SVM classifier on embeddings with downsampling embeddings of frequent gendered words, using all dimensions |
| `SVM_BERT_full_dimensions.ipynb` | Training an SVM classifier on embeddings with downsampling embeddings of frequent gendered words, once using one dimension and once all but the best dimension |
| `professions_sentences.ipynb` | Constructs "[gendered_word] is een [profession]" sentences for all gendered word–profession combinations, feeds them through the BERT model, and extracts the contextual embeddings for the profession in each sentence. Used to analyze how profession words are embedded in gendered contexts. |

