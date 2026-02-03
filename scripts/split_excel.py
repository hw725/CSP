import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define the dataset folder path
dataset_folder = "c:\\Users\\junto\\Downloads\\head-repo\\hw725\\CSP\\dataset"
output_folder = "c:\\Users\\junto\\Downloads\\head-repo\\hw725\\CSP\\dataset_split"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# List of Excel files to process
excel_files = ["paragraph.xlsx", "phrase.xlsx", "sentence.xlsx"]

# Split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

def split_and_save(file_path):
    # Read the Excel file
    df = pd.read_excel(file_path)

    # Ensure the '책명' column exists for splitting
    if '책명' not in df.columns:
        raise ValueError(f"The file {file_path} does not contain a '책명' column.")

    # Group by '책명' and split each group
    train_dfs, val_dfs, test_dfs = [], [], []
    for _, group in df.groupby('책명'):
        train, temp = train_test_split(group, test_size=(1 - train_ratio), random_state=42)
        val, test = train_test_split(temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)

    # Concatenate all groups back together
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)

    # Save the splits to new Excel files
    base_name = os.path.basename(file_path).replace('.xlsx', '')
    train_df.to_excel(os.path.join(output_folder, f"{base_name}_train.xlsx"), index=False)
    val_df.to_excel(os.path.join(output_folder, f"{base_name}_val.xlsx"), index=False)
    test_df.to_excel(os.path.join(output_folder, f"{base_name}_test.xlsx"), index=False)

# Process each Excel file
for excel_file in excel_files:
    file_path = os.path.join(dataset_folder, excel_file)
    split_and_save(file_path)