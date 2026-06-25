# HuggingFace Transformers for modeling
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# PyTorch for deep learning framework
import torch
from torch.utils.data import Dataset

# Scikit-learn for model evaluation and data splitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# NumPy and Pandas for data manipulation
import numpy as np
import pandas as pd

# Matplotlib for data visualization
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

print("All libraries loaded successfully")

# ==========================================
# Task 2: Load and Preprocess the Data
# ==========================================
from datasets import load_dataset

# 1. Load the dataset using HuggingFace datasets library
# We use a try-except block to first attempt loading the Webis dataset,
# and fallback to MRPC if it fails, as specified.
try:
    print("\nAttempting to load 'webis/sts-companion' dataset...")
    dataset = load_dataset("webis/sts-companion", split="train")
    print("Successfully loaded 'webis/sts-companion'")
except Exception as e:
    print(f"Could not load 'webis/sts-companion'. Falling back to 'glue', 'mrpc'...")
    dataset = load_dataset("glue", "mrpc", split="train")
    print("Successfully loaded 'glue' 'mrpc' dataset.")

# 2. Convert the dataset to a pandas DataFrame for easier exploration
df = pd.DataFrame(dataset)

print("\n--- Dataset Exploration ---")
# Print the first 5 rows
print("First 5 rows:")
print(df.head())

# Print the shape (number of rows and columns)
print(f"\nDataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Print the column names
print(f"\nColumn names: {df.columns.tolist()}")

# 3. Rename columns if needed to ensure we have standard names
# "sentence1" — first sentence, "sentence2" — second sentence, "label" — 0 or 1
rename_map = {}
if 'text1' in df.columns:
    rename_map['text1'] = 'sentence1'
if 'text2' in df.columns:
    rename_map['text2'] = 'sentence2'
if rename_map:
    df.rename(columns=rename_map, inplace=True)

# Value counts of the label column (how many paraphrases vs non-paraphrases)
print("\nLabel distribution (1 = paraphrase, 0 = non-paraphrase):")
print(df['label'].value_counts())

# 4. Drop any rows where sentence1 or sentence2 is null or empty
# First, replace empty strings with NaN so dropna() can catch them
df['sentence1'] = df['sentence1'].replace(r'^\s*$', np.nan, regex=True)
df['sentence2'] = df['sentence2'].replace(r'^\s*$', np.nan, regex=True)
initial_rows = len(df)
df.dropna(subset=['sentence1', 'sentence2'], inplace=True)
print(f"\nDropped {initial_rows - len(df)} rows with missing sentences.")

# 5. Print a final summary
num_paraphrases = (df['label'] == 1).sum()
num_non_paraphrases = (df['label'] == 0).sum()

print(f"\nDataset loaded: {len(df)} rows, {num_paraphrases} paraphrases, {num_non_paraphrases} non-paraphrases")

# ==========================================
# Task 3: Split the Dataset
# ==========================================

# 1. Extract the text data and labels from the DataFrame into lists
sentences1 = df['sentence1'].tolist()
sentences2 = df['sentence2'].tolist()
labels = df['label'].astype(int).tolist()

# WHY THREE SPLITS?
# 1. Train set: Used by the model to learn the patterns and update its weights.
# 2. Validation (Val) set: Used during training to monitor performance on unseen data, 
#    prevent overfitting, and help tune hyperparameters.
# 3. Test set: Kept completely hidden until the very end for an honest, unbiased 
#    evaluation of the final model's real-world performance.

# 2. First Split: Separate out 10% of the overall data as the TEST set.
# We stratify on labels to ensure the same ratio of paraphrases to non-paraphrases is maintained.
temp_s1, test_s1, temp_s2, test_s2, temp_labels, test_labels = train_test_split(
    sentences1, sentences2, labels,
    test_size=0.1,
    random_state=42,
    stratify=labels
)

# 3. Second Split: Separate the remaining 90% (temp) into TRAIN (80%) and VALIDATION (10%).
# 10% is 1/9th (approx 11.11%) of the remaining 90%.
train_s1, val_s1, train_s2, val_s2, train_labels, val_labels = train_test_split(
    temp_s1, temp_s2, temp_labels,
    test_size=1/9, # 11.11% of 90% is 10% of total
    random_state=42,
    stratify=temp_labels
)

# 5. Print the sizes of all three splits
print("\n--- Dataset Splits ---")
print(f"Train: {len(train_labels)} samples | Validation: {len(val_labels)} samples | Test: {len(test_labels)} samples")

# 6. Print the label distribution in each split to confirm stratification worked
print("\nLabel Distribution:")
print(f"Train labels      — 0: {train_labels.count(0)}, 1: {train_labels.count(1)}")
print(f"Validation labels — 0: {val_labels.count(0)}, 1: {val_labels.count(1)}")
print(f"Test labels       — 0: {test_labels.count(0)}, 1: {test_labels.count(1)}")

# ==========================================
# Task 4: Tokenize Datasets
# ==========================================

# WHY TOKENIZE?
# 1. RoBERTa (like all neural networks) only understands numbers, not raw text strings.
# 2. Tokenization converts text into numerical IDs.
# 3. For sentence pairs, RoBERTa needs special tokens to know where the first sentence 
#    ends and the second begins. Passing two lists together automatically adds the 
#    <s> sent1 </s></s> sent2 </s> format required by the model.

# 1. Load the pre-trained RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 2. Function to tokenize sentence pairs
def tokenize_pairs(sentences1, sentences2):
    return tokenizer(
        sentences1,
        sentences2,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

# 3. Call this function on all three splits
train_encodings = tokenize_pairs(train_s1, train_s2)
val_encodings = tokenize_pairs(val_s1, val_s2)
test_encodings = tokenize_pairs(test_s1, test_s2)

# 4. Print the shape to confirm it worked (should be num_samples x 128)
print("\n--- Tokenization ---")
print(f"Train encodings input_ids shape: {train_encodings['input_ids'].shape}")

# 5. Print one example before, during, and after tokenization
print("\nExample Tokenization (First pair in Train set):")
print(f"Original Sentence 1: {train_s1[0]}")
print(f"Original Sentence 2: {train_s2[0]}")

first_input_ids = train_encodings['input_ids'][0]
print(f"\nTokenized IDs: {first_input_ids}")

decoded_example = tokenizer.decode(first_input_ids)
print(f"\nDecoded back to text: {decoded_example}")

# ==========================================
# Task 5: Generate Tensors (PyTorch Dataset)
# ==========================================

# WHY DO WE NEED THIS CLASS?
# The HuggingFace Trainer requires data to be in a very specific format.
# By wrapping our encodings and labels in a PyTorch Dataset class, the Trainer 
# can automatically handle batching, shuffling, and moving data to the GPU.

class SimilarityDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # The Trainer expects a dictionary with 'input_ids', 'attention_mask', and 'labels'
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item

# 5. Instantiate all three datasets
train_dataset = SimilarityDataset(train_encodings, train_labels)
val_dataset = SimilarityDataset(val_encodings, val_labels)
test_dataset = SimilarityDataset(test_encodings, test_labels)

# 6. Test it works by printing properties
print("\n--- PyTorch Datasets ---")
print(f"Train Dataset length: {len(train_dataset)}")
print(f"Validation Dataset length: {len(val_dataset)}")
print(f"Test Dataset length: {len(test_dataset)}")

sample_item = train_dataset[0]
print(f"\nKeys in train_dataset[0]: {list(sample_item.keys())}")
print(f"Shape of input_ids in train_dataset[0]: {sample_item['input_ids'].shape}")

# ==========================================
# Task 6: Load Pretrained RoBERTa Model and Set Device
# Task 7: Prepare Training Arguments and Create Trainer Object
# ==========================================

# 1. Detect device automatically
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 2. Load the pretrained RoBERTa model for Sequence Classification
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=2
)
model.to(device)

# 3. Print model info
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params / 1_000_000:.1f}M")
print(f"Trainable parameters: {trainable_params / 1_000_000:.1f}M")

# 4. Write a compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# WHY THESE HYPERPARAMETERS?
# - learning_rate (2e-5): Small learning rates like 2e-5, 3e-5, or 5e-5 are standard 
#   for fine-tuning Transformers because the model is already pre-trained; we just 
#   want to gently nudge it towards our specific task without destroying its general knowledge.
# - warmup_steps (100): Gradually increases the learning rate from 0 to the target (2e-5) 
#   over 100 steps to prevent unstable training at the very beginning.
# - weight_decay (0.01): Adds a small penalty to the weights to prevent the model from 
#   becoming too complex and overfitting the training data.

# 5. Create TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy='epoch', # eval_strategy replaces evaluation_strategy in recent transformers
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

# 6. Create the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\nModel and Trainer ready. Starting training...")

# ==========================================
# Task 8: Train the Model
# ==========================================

# WHAT IS FINE-TUNING?
# We are NOT training a model from scratch. RoBERTa has already been pre-trained on 
# massive amounts of text and already understands the English language. 
# Fine-tuning means we take this pre-trained intelligence and gently adjust its weights 
# so it specializes in our specific task (detecting sentence similarity/paraphrases).

# 1. Start training
print("\n--- Training Started ---")
train_result = trainer.train()

# 2. Print completion metrics
metrics = train_result.metrics
train_time_mins = metrics["train_runtime"] / 60
print(f"\n--- Training Complete ---")
print(f"Total training time: {train_time_mins:.2f} minutes")
print(f"Final training loss: {metrics['train_loss']:.4f}")
print(f"Samples per second: {metrics['train_samples_per_second']:.2f}")

# 3. Save the trained model and tokenizer
model_dir = './roberta-similarity-model'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"\nModel and tokenizer saved to {model_dir}")

# 4. Extract training logs and plot history
log_history = trainer.state.log_history

# Separate out training loss and validation accuracy
train_steps = []
train_loss = []
val_epochs = []
val_accuracy = []

for log in log_history:
    if "loss" in log and "step" in log:
        train_steps.append(log["step"])
        train_loss.append(log["loss"])
    if "eval_accuracy" in log and "epoch" in log:
        val_epochs.append(log["epoch"])
        val_accuracy.append(log["eval_accuracy"])

# Create a figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Training loss over steps (line chart)
ax1.plot(train_steps, train_loss, color='steelblue', marker='o', markersize=4)
ax1.set_title("Training Loss Over Steps", fontsize=14)
ax1.set_xlabel("Steps")
ax1.set_ylabel("Loss")
ax1.grid(True, linestyle='--', alpha=0.7)

# Right: Validation accuracy over epochs (bar chart)
epochs_str = [f"Epoch {int(e)}" for e in val_epochs]
ax2.bar(epochs_str, val_accuracy, color='mediumseagreen')
ax2.set_title("Validation Accuracy Over Epochs", fontsize=14)
ax2.set_ylabel("Accuracy")
ax2.set_ylim([0, 1.0]) # Accuracy is between 0 and 1
ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved training history plot to 'training_history.png'.")
plt.show()

# 5. Print a summary table of metrics after each epoch
print("\n--- Validation Metrics Summary ---")
# Filter logs to just those that contain eval_accuracy
eval_logs = [log for log in log_history if "eval_accuracy" in log]
if eval_logs:
    df_evals = pd.DataFrame(eval_logs)
    # Select only the relevant columns for a clean summary table
    cols_to_show = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_runtime', 'eval_samples_per_second']
    # Ensure columns exist before filtering
    cols_present = [c for c in cols_to_show if c in df_evals.columns]
    print(df_evals[cols_present].to_string(index=False))

# ==========================================
# Task 9: Test the Model and Calculate Accuracy
# ==========================================

# METRICS EXPLAINED:
# - Accuracy: The percentage of total predictions that were correct. However, if 
#   90% of data is 'Not Paraphrase', a model that always guesses 'Not Paraphrase' 
#   gets 90% accuracy but is actually useless! That's why we need other metrics:
# - Precision: Out of all the times the model *claimed* it was a paraphrase, 
#   how many actually were? (Avoids False Positives).
# - Recall: Out of all the *actual* paraphrases in reality, how many did the 
#   model successfully find? (Avoids False Negatives).
# - F1 Score: The harmonic mean of Precision and Recall. It's a balanced grade 
#   that punishes extreme scores (e.g., high precision but terrible recall).

print("\n--- Evaluating on Test Set ---")
# 1. Run predictions on the unseen Test set
predictions = trainer.predict(test_dataset)
pred_logits = predictions.predictions
pred_labels = np.argmax(pred_logits, axis=1)
true_labels = predictions.label_ids

# 2. Calculate and print Accuracy and Classification Report
acc = accuracy_score(true_labels, pred_labels)
print(f"Overall Test Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=['Not Paraphrase', 'Paraphrase']))


# ==========================================
# Task 10: Compute and Display the Confusion Matrix
# ==========================================

# 3. Compute the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
tn, fp, fn, tp = cm.ravel()

# 4. Visualize it as a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(cm, interpolation='nearest', cmap='Blues')
fig.colorbar(cax)

classes = ['Not Paraphrase', 'Paraphrase']
tick_marks = np.arange(len(classes))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# Annotate each cell with its numerical count
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    ax.text(j, i, format(cm[i, j], 'd'),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=14)

ax.set_title("Confusion Matrix — RoBERTa Similarity Detection", fontsize=14)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix plot to 'confusion_matrix.png'.")
plt.show()

# 5. Plain-English interpretation
print("\n--- Confusion Matrix Breakdown ---")
print(f"True Positives (correctly identified paraphrases): {tp}")
print(f"True Negatives (correctly identified non-paraphrases): {tn}")
print(f"False Positives (wrongly called paraphrase): {fp}")
print(f"False Negatives (missed paraphrases): {fn}")


# ==========================================
# BONUS: Live Prediction Function
# ==========================================

def predict_similarity(sentence1, sentence2):
    # Prepare the inputs
    inputs = tokenizer(
        sentence1, 
        sentence2, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=128
    ).to(device)
    
    # Run through the model without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and apply softmax to get probabilities
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).squeeze()
    
    # Get the predicted class and its confidence
    pred_class = torch.argmax(probs).item()
    confidence = probs[pred_class].item() * 100
    
    label_str = "PARAPHRASE" if pred_class == 1 else "NOT PARAPHRASE"
    return f"{label_str} (Confidence: {confidence:.2f}%)"

print("\n--- BONUS: Testing Custom Pairs ---")

pair1 = ("The cat sat on the mat.", "A cat was resting on the rug.")
print(f"Pair 1: \n- {pair1[0]}\n- {pair1[1]}")
print(f"Result: {predict_similarity(*pair1)}\n")

pair2 = ("It is raining heavily.", "The stock market crashed today.")
print(f"Pair 2: \n- {pair2[0]}\n- {pair2[1]}")
print(f"Result: {predict_similarity(*pair2)}\n")

pair3 = ("She loves to read books.", "Reading is her favourite hobby.")
print(f"Pair 3: \n- {pair3[0]}\n- {pair3[1]}")
print(f"Result: {predict_similarity(*pair3)}\n")

print("Congratulations! The project is complete!")
