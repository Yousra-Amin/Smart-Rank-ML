import pandas as pd
import os

# Get project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cleaned_path = os.path.join(BASE_DIR, "data", "cleaned")

# Input file from Week 3
input_file = os.path.join(cleaned_path, "top1000_no_duplicates.csv")

print("Loading dataset...")

df = pd.read_csv(input_file)

print("Dataset shape before handling missing values:", df.shape)

# Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows where query or passage is missing
df = df.dropna(subset=["query", "passage"])

# Fill any remaining missing text values safely
df["query"] = df["query"].fillna("")
df["passage"] = df["passage"].fillna("")

print("\nDataset shape after handling missing values:", df.shape)

# Check missing values again
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Save cleaned dataset
output_file = os.path.join(cleaned_path, "top1000_no_missing.csv")
df.to_csv(output_file, index=False)

print("\nClean dataset saved successfully.")
print("File:", output_file)