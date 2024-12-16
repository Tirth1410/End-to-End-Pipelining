import json
import numpy as np
import torch
import torch.nn as nn
from azureml.core.model import Model
# from azureml.core import Workspace, Dataset
from azure.storage.blob import BlobServiceClient

# Load the vocab dictionary from `vocab.txt`
def load_vocab(vocab_file_path):
    with open(vocab_file_path, "r", encoding="utf-8") as f:
        vocab = {line.split()[0]: int(line.split()[1]) for line in f}
    # Ensure special tokens
    vocab.update({"<unk>": 0, "<pad>": 1, "<eos>": 2})
    return vocab

# Tokenize input text based on vocab
def tokenize_text(text, vocab, max_token_length=250):
    tokens = [vocab.get(word.lower(), vocab["<unk>"]) for word in text.split()]
    tokens = tokens[:max_token_length]
    tokens.append(vocab["<eos>"])

    return np.array(tokens)

# Initialize and load the model and vocab
def init():
    global model, vocab

    model_path=Model.get_model_path("new_model_state_dict.pth")

    class SAnalysis(nn.Module):
        def __init__(self,vocab_size):
            super().__init__()
            self.em=nn.Embedding(vocab_size,128)
            self.drop=nn.Dropout(0.2)
            self.gru=nn.GRU(128,256,batch_first=True)
            self.classifier=nn.Linear(256,2)

        def forward(self,x):
            x=self.em(x)
            x=self.drop(x)
            outputs,hidden=self.gru(x)
            hidden.squeeze_(0)
            x=self.classifier(hidden)

            return x
        
    model=SAnalysis(30124)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    
    # Define the connection string and container details
    connection_string = "<your-connection-string-here>"
    container_name = "<your-container-name-here>"
    blob_path = "vocab_data/vocab.txt"  #These can be changed according to your file structure and requirements.
    target_path = "./vocab.txt"         #(For Beginners) It is recommended that keep the paths same and stick to the readme file.

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

    # Download the blob
    with open(target_path, "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())

    # Load the vocabulary
    vocab = load_vocab(target_path)

# Run scoring based on tokenized input text
def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_text = data.get("input_text") #Take note of the json format
        
        # Tokenize and format as required by the model
        input_tokens = tokenize_text(input_text, vocab)
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0)  # Shape [1, max_token_length]
        
        # Make prediction
        with torch.no_grad():
            logits = model(input_tensor)
        
        # Convert logits to class prediction
        predicted_class = int(torch.argmax(logits, dim=1).item())
        
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}
