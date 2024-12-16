import pandas as pd
import numpy as np
import argparse
from azureml.core import Workspace, Dataset, Datastore
from sklearn.model_selection import train_test_split
import os

# Argument parsing for vocab and dataset file paths
parser = argparse.ArgumentParser(description="Process dataset and vocab for train/test split.")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset CSV file in Azure Blob Storage.")
parser.add_argument('--processed_data', type=str, required=True, help='Path to save model output')
args = parser.parse_args()

os.makedirs(args.processed_data, exist_ok=True)

print("Inside the script-1 file")

# Connect to the Azure workspace
workspace = Workspace(subscription_id="<your-subscription-id>",
                      resource_group="<your-resource-group>",
                      workspace_name="<your-workspace-name>")

print("Connected to workspace")

# Load dataset.csv from Azure Blob Storage
datastore = workspace.get_default_datastore()
dataset = Dataset.Tabular.from_delimited_files(path=(datastore, f"datasets/{args.dataset_name}"))
df = dataset.to_pandas_dataframe()

print("Got the dataset")

# Load vocab.txt from Azure Blob Storage
vocab_dataset = Dataset.File.from_files((datastore, f"vocab_data/vocab.txt"))
vocab_file_path = vocab_dataset.download(target_path=".", overwrite=True)[0]

print("Got the vocab")

with open(vocab_file_path, "r", encoding="utf-8") as f:
    vocab = {line.split()[0]: int(line.split()[1]) for line in f}

# Ensure special tokens "<unk>", "<pad>", and "<eos>" are in the vocab dictionary
special_tokens = {"<unk>": 0, "<pad>": 1, "<eos>": 2}
for token, index in special_tokens.items():
    vocab[token] = index
default_index = vocab["<unk>"]

# Tokenizer function using vocab dictionary
def tokenize_review(review, vocab, max_token_length=250):
    tokens = [vocab.get(word.lower(), default_index) for word in review.split()]
    tokens = tokens[:max_token_length]  # Limit to max_token_length
    tokens.append(vocab["<eos>"])  # Append end-of-sequence token
    if len(tokens) < max_token_length:
        tokens.extend([vocab["<pad>"]] * (max_token_length - len(tokens)))  # Pad if shorter
    return tokens

# Tokenize the reviews
df["tokens"] = df["review"].apply(lambda review: tokenize_review(review, vocab))

print("Tokenisation complete")

# Split dataset into features (X) and labels (Y)
X = np.array(df["tokens"].tolist())
Y = np.array(df["label"].tolist()).reshape(-1, 1)

# Split into train and test sets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Got traiin_x,train_y,test_x,test_y")

# Adjust dimensions to Reviews x 100 for X, Reviews x 1 for Y
train_x = train_x[:, :100]
test_x = test_x[:, :100]

# Save the train/test splits to workspace
np.save(os.path.join(args.processed_data, "train_x.npy"), train_x)
np.save(os.path.join(args.processed_data, "train_y.npy"), train_y)
np.save(os.path.join(args.processed_data, "test_x.npy"), test_x)
np.save(os.path.join(args.processed_data, "test_y.npy"), test_y)

files = [os.path.join(args.processed_data, "train_x.npy"),os.path.join(args.processed_data, "train_y.npy"),os.path.join(args.processed_data, "test_x.npy"),os.path.join(args.processed_data, "test_y.npy")]

# Define the target path in the datastore
target_directory = "processed_data/"

# Upload files individually
for file in files:
    # Define the full target path
    target_path = target_directory + os.path.basename(file)
    
    # Upload the file to the datastore
    datastore.upload_files(
        files=[file],  # List of files to upload
        target_path=target_directory,  # Target path in the datastore
        overwrite=True  # Overwrite if file already exists
    )

print("Processed data uploaded to blob storage")

print("Train/test splits saved to workspace as `traindata` and `testdata`.")
