# Phishing Email Detection using Fine-tuned RoBERTa

This project demonstrates the fine-tuning of the RoBERTa large model for detecting phishing emails. The model is fine-tuned on a dataset of phishing and legitimate emails and evaluated using various metrics.

## Project Structure

- `fine_tuning_script.py`: Script for fine-tuning the RoBERTa model on the phishing email dataset.
- `measurement_script.py`: Script for evaluating the fine-tuned model and calculating performance metrics.
- `Cleaned_Phishing_Email.csv`: Dataset file containing phishing and legitimate emails.
- `fine_tuned_phishing_roberta_large/`: Directory containing the fine-tuned model and tokenizer.
- `measurements-optimized-version/`: Directory containing the evaluation metrics file.

## Fine-tuning

The `fine_tuning_script.py` script performs the following steps:

1. Loads and preprocesses the dataset from `Cleaned_Phishing_Email.csv`.
2. Initializes the RoBERTa tokenizer and model.
3. Defines the training arguments and data collator.
4. Tokenizes and encodes the dataset.
5. Fine-tunes the model using the Trainer class from the Transformers library.
6. Saves the fine-tuned model and tokenizer to the `fine_tuned_phishing_roberta_large/` directory.

## Evaluation

The `measurement_script.py` script performs the following steps:

1. Loads the fine-tuned model and tokenizer from the `fine_tuned_phishing_roberta_large/` directory.
2. Prepares the dataset for evaluation.
3. Defines the evaluation arguments.
4. Initializes the Trainer class for evaluation.
5. Evaluates the model on the test dataset.
6. Calculates various performance metrics, including accuracy, precision, recall, F1 score, Matthews Correlation Coefficient, balanced accuracy, area under the precision-recall curve, and area under the ROC curve.
7. Writes the evaluation metrics to a file in the `measurements-optimized-version/` directory.

## Results

The evaluation results obtained from running the `measurement_script.py` are as follows:

```
Accuracy: 0.9891
Precision: 0.9730
Recall: 1.0000
F1 Score: 0.9863
Matthews Correlation Coefficient: 0.9775
Balanced Accuracy: 0.9910
Area Under the Precision-Recall Curve: 0.9730
Area Under the ROC Curve: 0.9910

Confusion Matrix:
[[11119, 203],
 [0, 7312]]
```

These results indicate that the fine-tuned RoBERTa model achieves high accuracy, precision, recall, and F1 score in detecting phishing emails. The confusion matrix shows that the model correctly classifies a significant portion of both phishing and legitimate emails.

## Usage

To run the fine-tuning script:
```
python fine_tuning_script.py
```

To run the evaluation script:
```
python measurement_script.py
```

Make sure to have the required dependencies installed and the dataset file `Cleaned_Phishing_Email.csv` in the same directory as the scripts.

## Dependencies

- Python 3.x
- PyTorch
- Transformers
- Datasets
- NumPy
- scikit-learn


# LLM-Phishing
# LLM-Phishing
