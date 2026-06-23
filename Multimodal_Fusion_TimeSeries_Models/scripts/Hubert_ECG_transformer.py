import torch
from HuBERT-ECG import (
    HubertECGForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import os
DATA_DIR = "/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/Multimodal_Fusion_TimeSeries_Models/data"

import sys

# Add the cloned repository's path to sys.path
repo_path = os.path.abspath("/Users/devleibowitz/Documents/TAU Courses/Deep Learning Raja/HW/dl_homeworks/Multimodal_Fusion_TimeSeries_Models/HuBERT-ECG")
sys.path.insert(0, repo_path)

# Now you can import the module
import HuBERT # Replace with the actual module name


# 1. LOAD PRETRAINED MODEL
model_name = "Edoardo-BS/hubert-ecg-large"  # Pretrained HuBERT-ECG model
config = HubertECGConfig.from_pretrained(model_name)
model = HubertECGForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    num_labels=2 # Set this to the number of output classes in your data
)

# 2. LOAD AND PROCESS INPUT DATA
label_path = os.path.join(DATA_DIR, "processed", "phenotype_2_labels.csv")
tensor_path = os.path.join(DATA_DIR, "processed", "fhr_uc_signal_tensor.pt ")

labels_df = pd.read_csv(label_path)  # Load CSV labels
signals_tensor = torch.load(tensor_path)  # Load FHR + UC tensor
# Load tensor signals (shape: [552, 21618, 2]) and extract the first signal (fetal heart rate)
signals = signals_tensor[:, :, 0]  # Extract the first signal (shape: [552, 21618])

# Load labels from phenotype_2_labels.csv
labels_df = pd.read_csv("phenotype_2_labels.csv")  # Ensure this file has columns 'sample' and 'label'
labels_dict = dict(zip(labels_df["sample"], labels_df["label"]))  # Create a dictionary mapping sample to label

# Match signals with labels based on sample indices
samples_indices = range(len(signals))  # Assuming samples are indexed sequentially from 0 to len(signals)-1
labels = [labels_dict[idx] for idx in samples_indices]  # Get labels corresponding to each sample

# Convert signals and labels into a Hugging Face Dataset format
data_dict = {
    "signals": signals.numpy(),  # Convert tensor to numpy array for compatibility
    "labels": np.array(labels)   # Labels as numpy array
}

dataset = Dataset.from_dict(data_dict)

# Preprocess function for HuBERT-ECG input format
def preprocess_function(examples):
    inputs = torch.tensor(examples["signals"])  # Convert signals to torch tensors (shape: [n_samples, sequence_length])
    return {"input_values": inputs}

processed_dataset = dataset.map(preprocess_function, batched=True)

# Split into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(processed_dataset))
train_data = processed_dataset.select(range(train_size))
val_data = processed_dataset.select(range(train_size, len(processed_dataset)))

# 3. TRAINING SETUP
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    logging_dir="./logs",
    fp16=True,  # Enable mixed precision for faster training
)

# 4. METRICS
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 5. TRAINER
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 6. TRAIN THE MODEL
trainer.train()

# 7. SAVE MODEL WEIGHTS (After training)
trainer.save_model("hubert_ecg_finetuned")
torch.save(model.state_dict(), "hubert_ecg_weights.pth")  # Save weights separately[1][3]

# 8. EVALUATION
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names=[f"Class {i}" for i in range(YOUR_NUM_CLASSES)]))
    
    # ROC Curve (for binary classification)
    if YOUR_NUM_CLASSES == 2:
        probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png")
        plt.close()
    
    return {"accuracy": (predictions == labels).mean()}

# Run final evaluation
results = trainer.evaluate()
print("Final evaluation results:", results)

# 9. PLOT TRAINING LOSS (Add to TrainingArguments)
training_args = TrainingArguments(
    # ... [previous arguments] ...
    logging_steps=10,  # Log every 10 steps
    report_to="none"
)

# Plot loss curve post-training
loss_values = trainer.state.log_history
train_loss = [x['loss'] for x in loss_values if 'loss' in x]
steps = [x['step'] for x in loss_values if 'loss' in x]

plt.figure()
plt.plot(steps, train_loss, label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()