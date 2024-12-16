import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from azureml.core import Workspace, Datastore, Dataset, Model
import joblib

parser = argparse.ArgumentParser(description="Process dataset and vocab for train/test split.")
parser.add_argument('--processed_data', type=str, help='Path to processed data')
parser.add_argument('--processed_data2', type=str, help='Path to save model output')
args = parser.parse_args()


os.makedirs(args.processed_data2, exist_ok=True)


# Connect to the Azure workspace
workspace = Workspace(subscription_id="<your-subscription-id>",
                      resource_group="<your-resource-group>",
                      workspace_name="<your-workspace-name>")

print("Connected to workspace")


datastore = Datastore.get(workspace, "workspaceblobstore")
# dataset = Dataset.File.from_files(path=(datastore, 'processed_data'))

# dataset.download(target_path='./', overwrite=True)

print("Downloaded the training data")

# Check the paths to confirm that files are downloaded
train_x_path = os.path.join(args.processed_data,"train_x.npy")
train_y_path = os.path.join(args.processed_data,"train_y.npy")
test_x_path = os.path.join(args.processed_data,"test_x.npy")
test_y_path = os.path.join(args.processed_data,"test_y.npy")

train_x = np.load(train_x_path)  # Corrected this to access the file path properly
train_y = np.load(train_y_path)  # Corrected this to access the file path properly
test_x = np.load(train_x_path)  # Corrected this to access the file path properly
test_y = np.load(train_y_path) 

# Convert train_x and train_y to PyTorch tensors
train_x_tensor = torch.tensor(train_x, dtype=torch.long)
train_y_tensor = torch.tensor(train_y, dtype=torch.long).squeeze()


models = Model.list(workspace)

# Find the model with the "deployed" tag set to "true"
deployed_models = [model for model in models if model.tags.get("deployed") == "true"]

model=deployed_models[0]

model.download(target_dir="./",exist_ok=True)

os.rename("./new_model_state_dict.pth","./model_state_dict.pth")

model_path = "./model_state_dict.pth"  # Use f-string for correct path

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

torch.save(model.state_dict(),os.path.join(args.processed_data2,"model_state_dict.pth"))

print("Loaded the model")

# Define loss function (CrossEntropyLoss) and optimizer (Adam)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model for 5 epochs
for epoch in range(5):
    
    model.train()  # Set the model to training mode
    

    # Forward pass
    output = model(train_x_tensor)  # Feed the train_x_tensor to the model

    loss = loss_function(output, train_y_tensor)  # Calculate the loss
    print("LOSS IS:",loss.item())
    
    # Backward pass
    optimizer.zero_grad()  # Zero gradients before backward pass
    loss.backward()  # Compute gradients
    optimizer.step()  # Update the weights using the optimizer

torch.save(model.state_dict(),os.path.join(args.processed_data2,"new_model_state_dict.pth"))
np.save(os.path.join(args.processed_data2, "test_x.npy"), test_x)
np.save(os.path.join(args.processed_data2, "test_y.npy"), test_y)


print("Training over!")
latest_version = max([int(model.tags.get("version")) for model in models], default=0)

print("The latest version of the model is ",latest_version)

model = Model.register(workspace=workspace,
                       model_path=os.path.join(args.processed_data2,"new_model_state_dict.pth"),  # Path to your model.pkl file
                       model_name="new_model_state_dict.pth",  # Choose a name for your model
                       tags={"version": f"{latest_version+1}", "deployed": "false"})

print("Registered the model")

