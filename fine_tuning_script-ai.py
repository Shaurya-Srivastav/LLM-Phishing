import os
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding, set_seed)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, matthews_corrcoef, balanced_accuracy_score

# Set environment variable to handle parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure reproducibility
set_seed(42)

print("Loading and preprocessing the dataset...")

# Function to encode labels in the dataset
def encode_labels(example):
    example['labels'] = int(example['Email Type'])
    return example

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    dataset = load_dataset('csv', data_files=file_path)['train']
    dataset = dataset.map(encode_labels)
    return DatasetDict(train=dataset.train_test_split(test_size=0.1)['train'],
                       test=dataset.train_test_split(test_size=0.1)['test'])

# Function for tokenization and encoding
def tokenize_and_encode(examples):
    tokenized_inputs = tokenizer(examples['Email Text'], padding=False, truncation=True, max_length=512)
    tokenized_inputs['labels'] = examples['labels']
    return tokenized_inputs

# Compute metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'mcc': mcc, 'balanced_acc': balanced_acc}

print("Initializing tokenizer and model...")

model_name = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

print("Defining training arguments...")

training_args = TrainingArguments(
    output_dir='./ai-results-agumentedd',
    num_train_epochs=6,
    per_device_train_batch_size=8,
    learning_rate=3e-5,
    warmup_ratio=0.06,
    weight_decay=0.01,
    evaluation_strategy='steps',
    eval_steps=100,
    logging_dir='./ai-logs-agumented',
    logging_steps=10,
    save_strategy='steps',
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    fp16=True,
    dataloader_num_workers=4,
    report_to="none",
)

print("Loading, preprocessing, tokenizing, and encoding the dataset...")

dataset_path = 'augument-ai-human-dataset.csv'  # Update this path
dataset = load_and_preprocess_data(dataset_path)
tokenized_datasets = dataset.map(tokenize_and_encode, batched=True, remove_columns=['Email Text', 'Email Type', 'labels'])

print("Fine-tuning the model...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("Saving the fine-tuned model and tokenizer...")

model_path = './augumented_phishing_roberta_large'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"Model and tokenizer have been fine-tuned and saved to {model_path}")
