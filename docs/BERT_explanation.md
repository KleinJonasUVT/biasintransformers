# Training an Encoder-Only Model with an MLM Head for Word Embeddings

This document outlines the methodology for training an encoder-only Transformer model equipped with a Masked Language Model (MLM) head, using the Hugging Face Transformers library. The structure explains the structure also used in the thesis. The primary objective is to extract meaningful word embeddings by training the model from scratch, ensuring that the learned representations are derived solely from the Dutch SONAR corpus.

## Model Structure

The training is done following the standard BERT architecture with an MLM head, as illustrated below:

![BERT MLM Overview](https://github.com/KleinJonasUVT/biasintransformers/blob/52def7c6f06ae26d41ca9c9c6e1aab9ea9d96c49/assets/images/colapsed_BERT.png)

Additionally, an expanded version of the visual representation is available, providing a more detailed breakdown of the model's components and the ability to reference specific parts of the model:

![Expanded BERT MLM Overview](https://github.com/KleinJonasUVT/biasintransformers/blob/52def7c6f06ae26d41ca9c9c6e1aab9ea9d96c49/assets/images/expanded.png)

The model consists of the following key components:
- **Tokenizer**: Responsible for preprocessing text by tokenizing sequences and adding special tokens such as `[CLS]`, `[MASK]`, and `[PAD]`.
- **BERT Encoder**: A Given Transformer architecture that encodes tokenized input into contextual representations.
- **MLM Head**: A feed-forward neural network tasked with predicting masked tokens within input sequences.
- **Output Logits**: Mapped to the vocabulary to be able to predict tokens during pretraining, and backpropagate with the loss function regarding these predictions.

## Random Initialization of Embeddings

As the model is trained from scratch, the embeddings are initialized randomly rather than starting from pretrained representations. This may be better visualized in the expanded visualisation of the structure.