import torch
import pandas as pd
import numpy as np
import os
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from matplotlib.colors import LinearSegmentedColormap

# Download tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Constants
MAX_LEN = 512
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 1e-4
METRICS = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability']

def load_and_prepare_data(data_path, tokenizer, metric_name):
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded dataframe with shape: {df.shape}")
    
    id2label = {0: "No", 2: "Yes"}
    label2id = {"No": 0, "Yes": 2}
    
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
    
    # Process data into expected format for seq2seq
    data_dict = {
        "train": {
            "input_text": [],
            "target_text": []
        },
        "validation": {
            "input_text": [],
            "target_text": []
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
    train_indices, val_indices = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=df[metric_name],
        random_state=42
    )
    
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    
    # Verify stratification worked
    print(f"Train set class distribution:")
    print(train_df[metric_name].value_counts())
    print(f"Validation set class distribution:")
    print(val_df[metric_name].value_counts())
    
    # Process train data with T5 formatting
    prefix = f"classify {metric_name}: "
    
    for _, row in train_df.iterrows():
        conversation = str(row['conversation_history'])
        response = str(row['tutor_response'])
        
        # For T5, format as question-answer pair
        input_text = prefix + f"{conversation} [SEP] {response}"
        
        # Convert the label ID to a text label
        label_id = get_label(row)
        target_text = id2label[label_id]
        
        data_dict["train"]["input_text"].append(input_text)
        data_dict["train"]["target_text"].append(target_text)
    
    # Process validation data
    for _, row in val_df.iterrows():
        conversation = str(row['conversation_history'])
        response = str(row['tutor_response'])
        
        input_text = prefix + f"{conversation} [SEP] {response}"
        
        label_id = get_label(row)
        target_text = id2label[label_id]
        
        data_dict["validation"]["input_text"].append(input_text)
        data_dict["validation"]["target_text"].append(target_text)
    
    # Create Hugging Face datasets
    train_dataset = Dataset.from_dict(data_dict["train"])
    val_dataset = Dataset.from_dict(data_dict["validation"])
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    # Tokenize the datasets
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True
        )
        
        # Setup the targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"], 
                max_length=8,
                padding="max_length",
                truncation=True
            )
            
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    
    return tokenized_datasets, id2label, label2id

def postprocess_text(preds, labels):
    # Strip the predictions and labels
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    
    return preds, labels

def compute_metrics(eval_preds):
    global tokenizer  # Use the global NLTK tokenizer
    
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Decode generated tokens
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in the labels (used for ignored positions) with pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    
    # Squish decoded_labels from list of lists to flat list
    decoded_labels = [l[0] for l in decoded_labels]
    
    # Map text back to IDs for evaluation
    label_map = {"No": 0, "To some extent": 2, "Yes": 2}
    
    # Normalize predictions to match exactly one of the expected values
    def normalize_prediction(pred):
        pred = pred.lower()
        if "no" in pred and "some" not in pred and "yes" not in pred:
            return "No"
        else:
            return "Yes"
    
    normalized_preds = [normalize_prediction(pred) for pred in decoded_preds]
    
    # Store for later use in confusion matrix
    global all_decoded_preds, all_decoded_labels
    all_decoded_preds = normalized_preds
    all_decoded_labels = decoded_labels
    
    # Map text predictions and labels to IDs
    pred_ids = [label_map.get(pred, 0) for pred in normalized_preds]
    label_ids = [label_map.get(label, 0) for label in decoded_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(label_ids, pred_ids)
    f1_weighted = f1_score(label_ids, pred_ids, average="weighted")
    f1_macro = f1_score(label_ids, pred_ids, average="macro")
    
    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro
    }

def train_model_for_metric(metric_name, data_path):
    global tokenizer, all_decoded_preds, all_decoded_labels
    all_decoded_preds, all_decoded_labels = [], []
    
    print(f"\n=== Training T5 model for {metric_name} ===")
    
    # Initialize tokenizer
    model_name = "t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load and prepare data
    tokenized_datasets, id2label, label2id = load_and_prepare_data(data_path, tokenizer, metric_name)
    if tokenized_datasets is None:
        print(f"Skipping {metric_name} due to data loading issues")
        return None
    
    # Initialize T5 model for classification (conditional generation)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Ensure output directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures/confusion-matrices", exist_ok=True)
    
    # Training arguments
    output_dir = f"models/t5_{metric_name.lower()}_model"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="f1_weighted",
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=8
    )
    
    # Create data collator for seq2seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        model=model,
        padding=True
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train the model
    print(f"Starting T5 training for {metric_name}...")
    trainer.train()
    
    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Evaluation results for T5 {metric_name} model:")
    print(eval_result)
    
    # Convert text predictions back to the IDs for confusion matrix and classification report
    label_map = {"No": 0, "To some extent": 1, "Yes": 2}
    pred_ids = [label_map.get(pred, 0) for pred in all_decoded_preds]
    label_ids = [label_map.get(label, 0) for label in all_decoded_labels]
    
    # Generate and print classification report
    print(f"\n=== Classification Report for {metric_name} ===")
    report = classification_report(
        y_true=label_ids, 
        y_pred=pred_ids,
        target_names=list(id2label.values()),
        labels=[0, 2],
        digits=4,
        zero_division=0
    )
    print(report)
    
    # Generate confusion matrix
    print(f"\n=== Confusion Matrix for {metric_name} ===")
    cm = confusion_matrix(
        y_true=label_ids, 
        y_pred=pred_ids
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
    cm_path = f"figures/confusion-matrices/enc-dec_{metric_name.lower()}_cm.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")
    
    # Save the model
    model_path = f"models/t5_{metric_name.lower()}_model_final"
    
    # Save evaluation results to a text file
    results_path = f"models/{metric_name.lower()}_evaluation.txt"
    with open(results_path, 'w') as f:
        f.write(f"=== Evaluation Results for T5 {metric_name} Model ===\n\n")
        f.write(f"Accuracy: {eval_result['eval_accuracy']:.4f}\n")
        f.write(f"F1 Weighted: {eval_result['eval_f1_weighted']:.4f}\n")
        f.write(f"F1 Macro: {eval_result['eval_f1_macro']:.4f}\n\n")
        f.write(f"=== Classification Report ===\n\n")
        f.write(report)
        f.write("\n\n=== Confusion Matrix ===\n\n")
        f.write(str(cm))
    print(f"Detailed evaluation results saved to {results_path}")
    
    return model_path

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Data path
    data_path = "data/clean_parsed_conversations.csv"
    
    # Train models for selected metrics
    # Note: T5 training will be slower than BERT due to the encoder-decoder architecture
    # yep, takes about 3 minutes per epoch on my 3060
    selected_metrics = ['Mistake_Identification', 'Actionability']
    
    model_paths = {}
    for metric in selected_metrics:
        model_path = train_model_for_metric(metric, data_path)
        if model_path:
            model_paths[metric] = model_path
    
    print("\n=== T5 Training Summary ===")
    print(f"Successfully trained T5 models for: {', '.join(model_paths.keys())}")

if __name__ == "__main__":
    main()