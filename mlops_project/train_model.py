# copy relevant code here and make sure models
# get saved in 
# C:\Users\anasn\OneDrive\Desktop\mlops\mlops_project\models


# change where models get saved

import torch
from models import model
from torch.utils.data import TensorDataset

from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

# load data an format it

data = torch.load(r"C:\Users\anasn\OneDrive\Desktop\mlops\data\processed\processed_img.pt")
labels = torch.load(r"C:\Users\anasn\OneDrive\Desktop\mlops\data\processed\processed_lab.pt")
train_set = TensorDataset(data, labels)

lr = .0001
model = model.MyAwesomeModel()

dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 5
#list to record trainig loss at each step
losses = []

for e in range(epochs):
    running_loss = 0
    for images, labels in dataloader:
        optimizer.zero_grad()
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)
        
#plot epochs against losses list

print(losses)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss at every epoch')
plt.grid(True)
#save the plot

figname = "training plot " + time.strftime("%Y-%m-%d %H-%M-%S") + ".png"
modelname = "model " + time.strftime("%Y-%m-%d %H-%M-%S") + ".pt"
plt.savefig(rf"C:\Users\anasn\OneDrive\Desktop\mlops\reports\figures\{figname}")

#save model using the checkpoint thing
torch.save(model, rf"C:\Users\anasn\OneDrive\Desktop\mlops\mlops_project\models\{modelname}")


print("Training done. Model and figure saved.")


