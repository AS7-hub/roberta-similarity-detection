# RoBERTa Sentence Similarity Detection

> Fine-tuned RoBERTa model for high-accuracy sentence similarity and paraphrase detection.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project implements a state-of-the-art Natural Language Processing (NLP) pipeline to determine whether two sentences are semantically equivalent (paraphrases). By fine-tuning the powerful `roberta-base` architecture on the Microsoft Research Paraphrase Corpus (MRPC), the model learns deep contextual representations of English sentences. This capability is essential for applications like duplicate question detection, semantic search, automated customer support, and text summarization, where understanding the nuanced meaning of sentences is critical.

## Key Results

The model achieves strong performance, proving its reliability in distinguishing semantic similarity. 

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | 83.38% |
| **Macro F1 Score** | 0.82 |
| **Training Time** | 4.15 minutes (NVIDIA T4 GPU) |
| **Model Parameters** | 124.6M |

## Folder Structure

```text
roberta-similarity-detection/
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignored files and folders
├── notebooks/                # Jupyter notebooks
│   ├── similarity_roberta.ipynb
│   └── similarity_roberta_demo.ipynb
├── src/                      # Source code scripts
│   ├── train.py              # Training logic
│   ├── predict.py            # Inference functions
│   └── app.py                # Gradio web interface
└── results/                  # Training metrics and charts
    ├── metrics_summary.md
    ├── confusion_matrix.png
    └── training_history.png
```

## Quick Start

### 1. Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/roberta-similarity-detection.git
cd roberta-similarity-detection
pip install -r requirements.txt
```

### 2. Exploring the Notebook

To view the complete end-to-end pipeline, including data exploration, training, and evaluation, you can run the primary Jupyter Notebook:

```bash
jupyter notebook notebooks/similarity_roberta.ipynb
```

### 3. Running Inference

You can easily use the model to predict the similarity of your own custom sentences using the `predict_similarity` function:

```python
from src.predict import predict_similarity

sentence1 = "The cat sat on the mat."
sentence2 = "A feline was resting on the rug."

# Output will indicate if it's a paraphrase and provide a confidence score
result = predict_similarity(sentence1, sentence2)
print(result) 
# Example Output: PARAPHRASE (Confidence: 94.50%)
```

## Dataset

This model was fine-tuned using the **Microsoft Research Paraphrase Corpus (MRPC)**. 
- **Total Pairs:** 3,668 sentence pairs
- **Source:** Extracted from news sources on the web, with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship.

## Model Architecture

The core of this project is the **`roberta-base`** model from HuggingFace. 
RoBERTa (Robustly Optimized BERT Pretraining Approach) builds on BERT's language masking strategy by removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates. This model was fine-tuned for sequence classification using the `Trainer` API from the HuggingFace `transformers` library.

## Performance Visualizations

### Confusion Matrix
The confusion matrix highlights the model's high true positive rate and reliability.
*(See `results/confusion_matrix.png` for the detailed visual)*

### Training History
The training history plot demonstrates stable loss convergence and consistent validation accuracy growth across epochs.
*(See `results/training_history.png` for the detailed visual)*

## License

This project is licensed under the MIT License.
