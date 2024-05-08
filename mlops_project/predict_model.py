# FOR NOW, THIS FILE ONLY TAKES IMAGES AS .PT FILES, FUTURE VERSIONS WILL HANDLE OTHER FILE FORMATS

import torch
import argparse
import sys
from torch.utils.data import DataLoader

# Create Argument Parser
parser = argparse.ArgumentParser(description='Example of command line arguments')

# Add Arguments
parser.add_argument('model', type=str, help='path of the model that will be used for prediction')
parser.add_argument('data', type=str, help='path of the data that will be used for prediction')

# Parse Arguments
args = parser.parse_args()

if not args.model.endswith('.pt'):
    print("Model file must be a .pt file")
    sys.exit(1)

if not args.data.endswith('.pt'):
    print("Data file must be a .pt file")
    sys.exit(1)

# Load the model
model = torch.load(args.model)

# Load the data
data = torch.load(args.data)
dataloader = DataLoader(data, batch_size=64, shuffle=True)




def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """


    return torch.cat([model(batch) for batch in dataloader], 0)

predictions = predict(model, dataloader)
print(type(predictions))
print(predictions)
print(predictions.size())
print(torch.exp(predictions))