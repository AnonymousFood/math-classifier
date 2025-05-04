import argparse
from sklearn.model_selection import train_test_split
import yaml
import torch
import pandas as pd
import numpy as np
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Constants
#MAX_LEN = 512
#MAX_LEN = 0
#BATCH_SIZE = 16
#EPOCHS = 5
#LEARNING_RATE = 2e-5
METRICS = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability'] # can only do first 4 metrics for now

def load_and_prepare_data(config, data_path, tokenizer, metric_name):
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded dataframe with shape: {df.shape}")
    
    # Create label mappings
    id2label = {0: "No", 1: "To some extent", 2: "Yes"}
    label2id = {"No": 0, "To some extent": 1, "Yes": 2}
    
    # Define label conversion function
    def get_label(row):
        if metric_name not in row:
            return 0
        val = row[metric_name]
        if pd.isna(val):
            return 0
        if val in label2id:
            return label2id[val]
        try:
            return int(val)
        except (ValueError, TypeError):
            print(f"Warning: Unknown value '{val}' in {metric_name}, using default 0")
            return 0
    
    # Process data into expected format
    data_dict = {
        "train": {
            "text": [],
            "labels": []
        },
        "validation": {
            "text": [],
            "labels": []
        }
    }
    
    # Check if metric exists in dataset
    if metric_name not in df.columns:
        print(f"Error: Metric '{metric_name}' not found in dataset")
        return None, None, None
    
    # Print unique values and counts
    print(f"Unique values in {metric_name}:")
    value_counts = df[metric_name].value_counts()
    print(value_counts)
    
    # Stratified split
    train_size = int(0.8 * len(df))
    train_indices, val_indices = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df[metric_name],
        random_state=42
    )

    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    
    # Process train data
    for _, row in train_df.iterrows():
        conversation = str(row['conversation_history'])
        response = str(row['tutor_response'])
        combined_text = f"[CONVERSATION] {conversation} [TUTOR_RESPONSE] {response}"
        data_dict["train"]["text"].append(combined_text)
        data_dict["train"]["labels"].append(get_label(row))
    
    # Process validation data
    for _, row in val_df.iterrows():
        conversation = str(row['conversation_history'])
        response = str(row['tutor_response'])
        combined_text = f"[CONVERSATION] {conversation} [TUTOR_RESPONSE] {response}"
        data_dict["validation"]["text"].append(combined_text)
        data_dict["validation"]["labels"].append(get_label(row))
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict(data_dict["train"])
    val_dataset = Dataset.from_dict(data_dict["validation"])
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Tokenize the datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            #max_length=MAX_LEN
            max_length=config['max_len']
        )
    
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    
    return tokenized_datasets, id2label, label2id

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    f1_macro = f1_score(labels, predictions, average="macro")
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro
    }

def train_model_for_metric(config, metric_name, data_path):
    """Train a model for a specific metric"""
    print(f"\n=== Training model for {metric_name} ===")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Load and prepare data
    tokenized_datasets, id2label, label2id = load_and_prepare_data(config, data_path, tokenizer, metric_name)
    if tokenized_datasets is None:
        print(f"Skipping {metric_name} due to data loading issues")
        return None
    
    # Initialize model 
    model = AutoModelForSequenceClassification.from_pretrained(
        #"bert-base-uncased", 
        config['model_name'], 
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id
    )
    
    # Training arguments
    output_dir = f"models/{metric_name.lower()}_model{config['index']}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        #learning_rate=LEARNING_RATE,
        #per_device_train_batch_size=BATCH_SIZE,
        #per_device_eval_batch_size=BATCH_SIZE,
        #num_train_epochs=EPOCHS,
        #weight_decay=0.01,
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        weight_decay=config['weight_decay'],
        #evaluation_strategy="epoch",
        #eval_strategy="epoch",
        #save_strategy="epoch",
        eval_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        push_to_hub=False,
    )
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print(f"Starting training for {metric_name}...")
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Evaluation results for {metric_name}:")
    print(eval_result)
    
    # Get predictions for detailed reporting
    val_predictions = trainer.predict(tokenized_datasets["validation"])
    preds = np.argmax(val_predictions.predictions, axis=1)
    labels = val_predictions.label_ids
    
    # Generate and print classification report
    print(f"\n=== Classification Report for {metric_name} ===")
    report = classification_report(
        y_true=labels, 
        y_pred=preds,
        target_names=list(id2label.values()),
        digits=4
    )
    print(report)
    
    # Generate confusion matrix
    print(f"\n=== Confusion Matrix for {metric_name} ===")
    cm = confusion_matrix(
        y_true=labels, 
        y_pred=preds
    )
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=list(id2label.values()), 
        yticklabels=list(id2label.values())
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {metric_name}')
    plt.tight_layout()
    
    # Save the confusion matrix plot
    cm_path = f"figures/confusion-matrices/enc_{metric_name.lower()}_cm.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save the model
    model_path = f"models/{metric_name.lower()}_model_final"
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model for {metric_name} saved to {model_path}")

    # Generate confusion matrix
    pred_output = trainer.predict(trainer.eval_dataset)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    y_true = pred_output.label_ids
    cm = confusion_matrix(y_true, y_pred)
    id2label = {0: "No", 1: "To some extent", 2: "Yes"}
    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    labels = list(id2label.values())
    labels = [id2label[i] for i in unique_labels]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"figures/tuned/confusion_matrix_{metric_name}_{config['index']}.png")

    # Generate classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        digits=4
    )

    with open(f"figures/tuned/classification_report_{metric_name}_{config['index']}.txt", "w") as f:
        f.write(report)
    
    return model_path

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Data path
    data_path = "data/clean_parsed_conversations.csv"
    
    # Train models for selected metrics (takes a little over 2 minutes per metric with my 3060 GPU)
    selected_metrics = ['Mistake_Identification', 'Actionability']

    # Mass testing file configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the config')
    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    
    model_paths = {}
    for metric in selected_metrics:
        model_path = train_model_for_metric(config, metric, data_path)
        if model_path:
            model_paths[metric] = model_path
    
    print("\n=== Training Summary ===")
    print(f"Successfully trained models for: {', '.join(model_paths.keys())}")
    print("Model Hyperparams")
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
