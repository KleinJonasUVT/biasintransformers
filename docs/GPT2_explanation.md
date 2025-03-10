# Training a Decoder-Only Model with an Autoregressive LM Head for Word Embeddings

This document outlines the methodology for training a decoder-only Transformer model equipped with an Autoregressive Language Modeling (LM) head, using the Hugging Face Transformers library. The primary objective is to extract meaningful word embeddings by training the model from scratch, ensuring that the learned representations are derived solely from the Dutch SONAR corpus.

## Model Structure

The training follows the standard GPT-2 architecture with an autoregressive LM head, as illustrated below:

<img src="https://github.com/KleinJonasUVT/biasintransformers/blob/52def7c6f06ae26d41ca9c9c6e1aab9ea9d96c49/assets/images/colapsed_GPT2.png" width="350"/>

Additionally, an expanded version of the visual representation, providing a more detailed breakdown of the model's components and the ability to reference specific parts of the model:

<img src="https://github.com/KleinJonasUVT/biasintransformers/blob/52def7c6f06ae26d41ca9c9c6e1aab9ea9d96c49/assets/images/GPT2.png" width="600"/>

The model consists of the following key components:
- **Tokenizer**: Processes input text using Byte Pair Encoding (BPE), converting it into subword tokens while maintaining spacing explicitly (e.g., `" GPT-2"` tokenizes as `["ĠGPT", "-", "2"]`).
- **GPT-2 Decoder**: A stack of Transformer decoder blocks, each with self-attention and feed-forward layers
- **LM Head**: A single linear layer that maps the final hidden states to vocabulary logits, allowing the model to predict the next token at each step.
- **Output Logits**: The model generates a probability distribution over the vocabulary, selecting the next token iteratively during training.

## Random Initialization of Embeddings

As the model is trained from scratch, the embeddings are initialized randomly rather than starting from pretrained representations. This may be better visualized in the expanded visualisation of the structure.