import pandas as pd
import os

# Get project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

# File path
file_path = os.path.join(cleaned_path, "top1000_sample.csv")

print("Loading sample dataset...")

df = pd.read_csv(file_path)

print("Dataset shape before cleaning:", df.shape)

# Check duplicate rows
duplicates = df.duplicated(subset=["qid", "pid"]).sum()

print("Number of duplicate rows:", duplicates)

# Remove duplicates
df_clean = df.drop_duplicates(subset=["qid", "pid"])

print("Dataset shape after removing duplicates:", df_clean.shape)

# Save cleaned dataset
output_file = os.path.join(cleaned_path, "top1000_no_duplicates.csv")

df_clean.to_csv(output_file, index=False)

print("\nClean dataset saved successfully.")
print("File:", output_file)