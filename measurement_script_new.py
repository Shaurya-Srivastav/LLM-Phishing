import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score,
                             matthews_corrcoef, balanced_accuracy_score, precision_recall_curve, average_precision_score,
                             roc_curve, classification_report)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def main(model_path='augumented_phishing_roberta_large', dataset_path='Cleaned_Phishing_Email.csv'):
    measurements_dir = "measurements-human-text"
    os.makedirs(measurements_dir, exist_ok=True)
    metrics_file = os.path.join(measurements_dir, "evaluation_metrics.txt")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print("Preparing dataset...")
    raw_dataset = load_dataset('csv', data_files=dataset_path)['train']
    processed_dataset = raw_dataset.map(lambda examples: preprocess_data(examples, tokenizer), batched=True)
    processed_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir='./tmp',
        per_device_eval_batch_size=8,
        report_to="none",
        no_cuda=False
    )

    print("Initializing the Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, metrics_file, measurements_dir)
    )

    print("Evaluating the model...")
    eval_results = trainer.evaluate(processed_dataset)
    print(f"Evaluation Results: {eval_results}")

def preprocess_data(examples, tokenizer):
    label_map = {'0': 0, '1': 1}
    result = tokenizer(examples['Email Text'], padding=True, truncation=True, max_length=512)
    labels = examples['Email Type']
    if isinstance(labels[0], int):
        labels = [str(label) for label in labels]
    result["labels"] = [label_map[label] for label in labels]
    return result

def compute_metrics(eval_pred, metrics_file, measurements_dir):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    mcc = matthews_corrcoef(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    auprc = average_precision_score(labels, predictions)
    auroc = roc_auc_score(labels, predictions)

    with open(metrics_file, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Area Under the Precision-Recall Curve: {auprc:.4f}\n")
        f.write(f"Area Under the ROC Curve: {auroc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(labels, predictions), separator=', '))
        f.write("\nClassification Report:\n")
        f.write(classification_report(labels, predictions, target_names=['Legitimate', 'Phishing']))

    # Generate confusion matrix plot
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(labels, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Legitimate', 'Phishing'], rotation=45)
    plt.yticks(tick_marks, ['Legitimate', 'Phishing'])
    thresh = cm.max() / 2.
    for i, j in [(i, j) for i in range(cm.shape[0]) for j in range(cm.shape[1])]:
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(measurements_dir, 'confusion_matrix.png'))
    plt.close()

    # Generate precision-recall curve
    precision, recall, _ = precision_recall_curve(labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig(os.path.join(measurements_dir, 'precision_recall_curve.png'))
    plt.close()

    # Generate ROC curve
    fpr, tpr, _ = roc_curve(labels, predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(os.path.join(measurements_dir, 'roc_curve.png'))
    plt.close()

    print(f"Metrics written to {metrics_file}")
    print("Graphs and charts saved to the measurements directory.")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'balanced_accuracy': balanced_acc,
        'auprc': auprc,
        'auroc': auroc
    }

if __name__ == "__main__":
    main()
