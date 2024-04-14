import pandas as pd

# Load the dataset
dataset_path = 'Phishing_Email.csv'
df = pd.read_csv(dataset_path)

# Remove the unnamed index column
df.drop(df.columns[0], axis=1, inplace=True)

# Encode the "Email Type" labels into numerical format
# "Safe Email" -> 0, "Phishing Email" -> 1
df['Email Type'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})

# Remove all rows where any column has a None/NaN value
df.dropna(inplace=True)

# Save the cleaned dataset to a new CSV file in the same directory
cleaned_dataset_path = 'Cleaned_Phishing_Email.csv'
df.to_csv(cleaned_dataset_path, index=False)

print("The dataset has been cleaned and saved to:", cleaned_dataset_path)

