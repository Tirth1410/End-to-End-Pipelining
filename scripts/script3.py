import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
from azureml.core import Workspace, Datastore, Dataset, Model
import joblib

parser = argparse.ArgumentParser(description="Process dataset and vocab for train/test split.")
parser.add_argument('--processed_data2', type=str, help='Path to save model output')
args = parser.parse_args()

# Connect to the Azure workspace
workspace = Workspace(subscription_id="<your-subscription-id>",
                      resource_group="<your-resource-group>",
                      workspace_name="<your-workspace-name>")

print("Connected to workspace")

# Check the paths to confirm that files are downloaded
test_x_path = os.path.join(args.processed_data2,"test_x.npy")
test_y_path = os.path.join(args.processed_data2,"test_y.npy")

test_x = np.load(test_x_path)  # Corrected this to access the file path properly
test_y = np.load(test_y_path)  # Corrected this to access the file path properly

print("Got the test data")

# Convert train_x and train_y to PyTorch tensors
test_x_tensor = torch.tensor(test_x, dtype=torch.long)
test_y_tensor = torch.tensor(test_y, dtype=torch.long).squeeze()

models = Model.list(workspace)

latest_version = max([int(model.tags.get("version")) for model in models], default=0)
ind1,ind2=0,0

for i in range(len(models)):
    if (int(models[i].tags.get("version"))==latest_version):
        ind1=i
    if (models[i].tags.get("deployed")=="true"):
        ind2=i

print("Got the indices of the models from models list")

# model1=models[ind1]
# model2=models[ind2]
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
  
model1=SAnalysis(30124)
# model1.download(target_dir="./", overwrite=True)
# model1_path = f"./{model1.name}"  # Use f-string for correct path
model1.load_state_dict(torch.load(os.path.join(args.processed_data2,"new_model_state_dict.pth")))  # Load the PyTorch model correctly
# model2.download(target_dir="./", overwrite=True)
# model2_path = f"./{model2.name}"  # Use f-string for correct path
model2=SAnalysis(30124)
model2.load_state_dict(torch.load(os.path.join(args.processed_data2,"model_state_dict.pth")))  # Load the PyTorch model correctly

print("loaded the models")


loss_function = nn.CrossEntropyLoss()

output1 = model1(test_x_tensor)  # Feed the train_x_tensor to the model
output2 = model2(test_x_tensor)  # Feed the train_x_tensor to the model

loss1 = loss_function(output1, test_y_tensor)  # Calculate the loss
loss2 = loss_function(output2, test_y_tensor)  # Calculate the loss

print("Found the losses of both the models")

if (loss1<=loss2):
    for model in models:
        if(int(model.tags.get("version"))==latest_version):
            model.update(tags={"version":f"{latest_version}","deployed":"true"})
            continue
        if (model.tags.get("deployed")=="true"):
            version=int(model.tags.get("version"))
            model.update(tags={"version":f"{version}","deployed":"false"})
            continue

print("Updated the tags")


#IF WE NEED TO USE THE OUTPUT LATER MAYBE WE CAN CHANGE THE PIPELINES A BIT FOR THAT
with open(os.path.join(args.processed_data2,"ci_output_status.txt"), "w") as f:
    f.write(str((loss1<=loss2).item()))

# Get the default datastore for the workspace
datastore = workspace.get_default_datastore()

# Upload the vocab.txt file to the datastore
datastore.upload_files(files=[os.path.join(args.processed_data2,"ci_output_status.txt")],
                       target_path="outputs/",  # Folder in the datastore where the file will be stored
                       overwrite=True)

print("Gave the result in a txt file")
