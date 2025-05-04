import argparse
import yaml
import torch
import pandas as pd
import numpy as np
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from huggingface_hub import login


from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict
import evaluate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from tqdm import tqdm


# Constants
#MAX_LEN = 512
#MAX_LEN = 0
#BATCH_SIZE = 16
#EPOCHS = 5
#LEARNING_RATE = 2e-5

METRICS = ['Mistake_Identification', 'Mistake_Location', 'Providing_Guidance', 'Actionability'] # can only do first 4 metrics for now

#def load_and_prepare_data(config, data_path, tokenizer, metric_name):
#    print(f"Loading data from: {data_path}")
#    df = pd.read_csv(data_path)
#    print(f"Loaded dataframe with shape: {df.shape}")
#    
#    # Create label mappings
#    id2label = {0: "No", 1: "To some extent", 2: "Yes"}
#    label2id = {"No": 0, "To some extent": 1, "Yes": 2}
#    
#    # Define label conversion function
#    def get_label(row):
#        if metric_name not in row:
#            return 0
#        val = row[metric_name]
#        if pd.isna(val):
#            return 0
#        if val in label2id:
#            return label2id[val]
#        try:
#            return int(val)
#        except (ValueError, TypeError):
#            print(f"Warning: Unknown value '{val}' in {metric_name}, using default 0")
#            return 0
#    
#    # Process data into expected format
#    data_dict = {
#        "train": {
#            "text": [],
#            "labels": []
#        },
#        "validation": {
#            "text": [],
#            "labels": []
#        }
#    }
#    
#    # Check if metric exists in dataset
#    if metric_name not in df.columns:
#        print(f"Error: Metric '{metric_name}' not found in dataset")
#        return None, None, None
#    
#    # Print unique values and counts
#    print(f"Unique values in {metric_name}:")
#    value_counts = df[metric_name].value_counts()
#    print(value_counts)
#    
#    # Train/validation split
#    train_size = int(0.8 * len(df))
#    train_df = df.iloc[:train_size]
#    val_df = df.iloc[train_size:]
#    
#    # Process train data
#    for _, row in train_df.iterrows():
#        conversation = str(row['conversation_history'])
#        response = str(row['tutor_response'])
#        combined_text = f"[CONVERSATION] {conversation} [TUTOR_RESPONSE] {response}"
#        data_dict["train"]["text"].append(combined_text)
#        data_dict["train"]["labels"].append(get_label(row))
#    
#    # Process validation data
#    for _, row in val_df.iterrows():
#        conversation = str(row['conversation_history'])
#        response = str(row['tutor_response'])
#        combined_text = f"[CONVERSATION] {conversation} [TUTOR_RESPONSE] {response}"
#        data_dict["validation"]["text"].append(combined_text)
#        data_dict["validation"]["labels"].append(get_label(row))
#    
#    # Create Hugging Face datasets
#    train_dataset = Dataset.from_dict(data_dict["train"])
#    val_dataset = Dataset.from_dict(data_dict["validation"])
#    dataset_dict = DatasetDict({
#        "train": train_dataset,
#        "validation": val_dataset
#    })
#    
#    # Tokenize the datasets
#    def tokenize_function(examples):
#        return tokenizer(
#            examples["text"],
#            padding="max_length",
#            truncation=True,
#            #max_length=MAX_LEN
#            max_length=config['max_len']
#        )
#    
#    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
#    
#    return tokenized_datasets, id2label, label2id
#
#def compute_metrics(eval_pred):
#    predictions, labels = eval_pred
#    predictions = np.argmax(predictions, axis=1)
#    
#    # Calculate metrics
#    accuracy = accuracy_score(labels, predictions)
#    f1_weighted = f1_score(labels, predictions, average="weighted")
#    f1_macro = f1_score(labels, predictions, average="macro")
#    
#    return {
#        "accuracy": accuracy,
#        "f1_weighted": f1_weighted,
#        "f1_macro": f1_macro
#    }


def train_model_for_metric(config, metric_name, data_path):
    print(f"\n=== Training decoder-only model for {metric_name} ===")


    # === Model & Tokenizer Setup ===
    model_name = config.get("model_name", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "llama" in model_name.lower() else torch.float32,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    # Create necessary directories for outputs
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/confusion-matrices", exist_ok=True)

    # === Prepare Dataset ===
    df = pd.read_csv(data_path)
    label_text_map = {
        "No": "No",
        "To some extent": "To some extent",
        "Yes": "Yes",
        0: "No", 1: "To some extent", 2: "Yes"
    }

    def prepare_row(row):
        prompt = f"[CONVERSATION] {row['conversation_history']} [TUTOR_RESPONSE] {row['tutor_response']}"
        target = label_text_map.get(row[metric_name], "No")
        return {"prompt": prompt, "target": target}

    data = df[df[metric_name].notna()].apply(prepare_row, axis=1)
    prompts = data.apply(lambda x: f"{x['prompt']}\nAnswer: {x['target']}")
    targets = data.apply(lambda x: x["target"])

    dataset = Dataset.from_dict({"text": prompts.tolist(), "target_text": targets.tolist()})
    dataset = dataset.train_test_split(test_size=0.2)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config.get("max_len", 128)
        )

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # === Training Setup ===
    training_args = TrainingArguments(
        output_dir=f"models/{metric_name}_{model_name.replace('/', '_')}",
        per_device_train_batch_size=config.get("batch_size", 2),
        per_device_eval_batch_size=config.get("batch_size", 2),
        num_train_epochs=config.get("epochs", 5),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=0.01,
        save_strategy="no",
        eval_strategy="steps",
        load_best_model_at_end=False,
        logging_dir="./logs",
        logging_steps=100,
        logging_first_step=True,
        report_to="all",
        eval_steps=200, # Evaluate every 20 steps
        logging_strategy="steps"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # === Training ===
    trainer.train()
    save_path = f"models/{metric_name}_{model_name.replace('/', '_')}_final"
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)
    # print(f"Model for {metric_name} (decoder-only) saved.")

    # === Evaluation ===
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    #eval_dataset = tokenized_dataset["test"]
    eval_dataset = tokenized_dataset["test"].remove_columns(["target_text"])
    raw_targets = dataset["test"]["target_text"]

    eval_loader = DataLoader(
        #tokenized_dataset["test"].add_column("target_text", dataset["test"]["target_text"]),
        eval_dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    )
    generated_preds = []
    true_labels = []
    for i, batch in enumerate(tqdm(eval_loader)):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=config.get("max_len", 128)
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_preds.extend(decoded)

        # Use the original text labels
        start = i * config.get("batch_size", 2)
        end = start + len(decoded)
        true_labels.extend(raw_targets[start:end])


    #label2id = {"No": 0, "To some extent": 1, "Yes": 2}
    label2id = {"No": 0, "Yes": 2}
    id2label = {v: k for k, v in label2id.items()}

    def map_text_to_label(text):
        text = text.strip().lower()
        if "no" in text and "some" not in text and "yes" not in text:
            return label2id["No"]
        else:
            return label2id["Yes"]

    y_pred = [map_text_to_label(p) for p in generated_preds]
    y_true = [map_text_to_label(t) for t in true_labels]

    cm = confusion_matrix(y_true, y_pred)
    unique_labels = sorted(set(y_true + y_pred))
    labels = [id2label[i] for i in unique_labels]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Before saving the confusion matrix
    cm_path = f"figures/confusion-matrices/dec_{metric_name.lower()}_cm.png"

    # Make sure parent directory exists
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)

    # Then save the plot
    plt.savefig(cm_path)

    os.makedirs("figures/tuned", exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
    with open(f"figures/tuned/classification_report_{metric_name}_{config['index']}.txt", "w") as f:
        f.write(report)

    return save_path



#def train_model_for_metric(config, metric_name, data_path):
#    print(f"\n=== Training decoder-only model for {metric_name} ===")
#
#    model_name = config.get("model_name", "gpt2")  # now configurable
#
#    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
#    if tokenizer.pad_token is None:
#        tokenizer.pad_token = tokenizer.eos_token
#    tokenizer.padding_side = "left" if "llama" in model_name.lower() else "right"
#
#    model = AutoModelForCausalLM.from_pretrained(
#        model_name,
#        torch_dtype=torch.float16 if "llama" in model_name.lower() else torch.float32,
#        device_map="auto"
#    )
#    model.resize_token_embeddings(len(tokenizer))
#
#    # Load and preprocess data
#    df = pd.read_csv(data_path)
#    label_text_map = {
#        "No": "No",
#        "To some extent": "To some extent",
#        "Yes": "Yes",
#        0: "No",
#        1: "To some extent",
#        2: "Yes"
#    }
#
#    def prepare_row(row):
#        prompt = f"[CONVERSATION] {row['conversation_history']} [TUTOR_RESPONSE] {row['tutor_response']}"
#        target = label_text_map.get(row[metric_name], "No")
#        return {"prompt": prompt, "target": target}
#
#    data = df[df[metric_name].notna()].apply(prepare_row, axis=1)
#    prompts = data.apply(lambda x: f"{x['prompt']}\nAnswer: {x['target']}")
#
#    dataset = Dataset.from_dict({"text": prompts.tolist()})
#    dataset = dataset.train_test_split(test_size=0.2)
#
#    def tokenize(batch):
#        return tokenizer(
#            batch["text"],
#            padding="max_length",
#            truncation=True,
#            max_length=config.get("max_len", 128)
#        )
#
#    tokenized_dataset = dataset.map(tokenize, batched=True)
#
#    training_args = TrainingArguments(
#        output_dir=f"models/{metric_name}_{model_name.replace('/', '_')}",
#        per_device_train_batch_size=config.get("batch_size", 2),
#        per_device_eval_batch_size=config.get("batch_size", 2),
#        num_train_epochs=config.get("epochs", 5),
#        learning_rate=config.get("learning_rate", 5e-5),
#        weight_decay=0.01,
#        #evaluation_strategy="steps",
#        eval_strategy="steps",
#        save_strategy="steps",
#        save_total_limit=1,
#        load_best_model_at_end=True,
#        logging_dir="./logs",
#        logging_steps=10,
#    )
#
#    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
#
#    trainer = Trainer(
#        model=model,
#        args=training_args,
#        train_dataset=tokenized_dataset["train"],
#        eval_dataset=tokenized_dataset["test"],
#        tokenizer=tokenizer,
#        data_collator=data_collator,
#    )
#
#    trainer.train()
#    save_path = f"models/{metric_name}_{model_name.replace('/', '_')}_final"
#    model.save_pretrained(save_path)
#    tokenizer.save_pretrained(save_path)
#
#    print(f"Model for {metric_name} (decoder-only) saved.")
#
#    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
#
#    def tokenize(batch):
#        return tokenizer(
#            batch["text"],
#            padding="max_length",     # ✅ ensures equal length
#            truncation=True,
#            max_length=config.get("max_len", 128)
#        )
#
#    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
#
#    # Free memory
#    model.eval()
#    torch.cuda.empty_cache()
#    torch.cuda.ipc_collect()  # optional: clears interprocess caching
#
#    # Evaluate
#    #pred_output = trainer.predict(trainer.eval_dataset)
#    pred_output = trainer.predict(
#        trainer.eval_dataset,
#        metric_key_prefix="eval",
#        ignore_keys=["logits"]  # 👈 prevent saving logits
#    )
#
#    #decoded_preds = tokenizer.batch_decode(pred_output.predictions, skip_special_tokens=True)
#    #decoded_labels = tokenizer.batch_decode(pred_output.label_ids, skip_special_tokens=True)
#
#    eval_loader = DataLoader(
#        tokenized_dataset["test"],
#        batch_size=config.get("batch_size", 2),
#        shuffle=False,
#        collate_fn=data_collator
#    )
#
#    generated_preds = []
#    true_labels = []
#
#    model.eval()
#    with torch.no_grad():
#        for batch in tqdm(eval_loader):
#            input_ids = batch["input_ids"].to(model.device)
#            attention_mask = batch["attention_mask"].to(model.device)
#
#            generated_ids = model.generate(
#                input_ids=input_ids,
#                attention_mask=attention_mask,
#                #max_length=config.get("max_len", 128)
#                max_new_tokens=config.get("max_len", 128)
#            )
#            outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
#            labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
#
#            generated_preds.extend(outputs)
#            true_labels.extend(labels)
#
#
#    id2label = {0: "No", 1: "To some extent", 2: "Yes"}
#    # Map generated text back to labels
#    def map_text_to_label(text):
#        text = text.strip().lower()
#        if "yes" in text:
#            return label2id["Yes"]
#        elif "no" in text:
#            return label2id["No"]
#        elif "some" in text or "extent" in text:
#            return label2id["To some extent"]
#        else:
#            return -1  # Unknown
#
#    #y_pred = [map_text_to_label(pred) for pred in decoded_preds]
#    #y_true = [map_text_to_label(label) for label in decoded_labels]
#    y_pred = [map_text_to_label(p) for p in generated_preds]
#    y_true = [map_text_to_label(t) for t in true_labels]
#
#    cm = confusion_matrix(y_true, y_pred)
#    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
#    labels = [id2label[i] for i in unique_labels]
#    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#    disp.plot(cmap='Blues', xticks_rotation=45)
#    plt.title("Confusion Matrix")
#    plt.tight_layout()
#    plt.savefig(f"figures/tuned/confusion_matrix_{metric_name}_{config['index']}.png")
#
#    report = classification_report(y_true, y_pred, target_names=labels, digits=4)
#    with open(f"figures/tuned/classification_report_{metric_name}_{config['index']}.txt", "w") as f:
#        f.write(report)
#
#    return save_path



def main():
    load_dotenv()  # loads .env from current working dir

    token = os.getenv("HUGGINGFACE_TOKEN")
    login(token)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Data path
    data_path = "data/clean_parsed_conversations.csv"
    
    # Choosen Metrics
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
            model_paths[metric] = "Completed" # don't save the model!
    
    print("\n=== Training Summary ===")
    print(f"Successfully trained models for: {', '.join(model_paths.keys())}")
    print("Model Hyperparams")
    for key, value in config.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
