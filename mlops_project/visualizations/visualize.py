import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

# Loads a pre-trained network

model = torch.load(r"C:\Users\anasn\OneDrive\Desktop\mlops\mlops_project\models\model 2024-05-07 13-40-50.pt")


# load data
data = r"C:\Users\anasn\OneDrive\Desktop\mlops\images10.pt"
dataloader = DataLoader(data, batch_size=64, shuffle=True)

# Extracts some intermediate representation of the data
feature_extractor = nn.Sequential(
    *list(model.children())[:3]  # Extracts layers up to the third convolutional layer
)

intermediate_representations = feature_extractor(dataloader)
# Apply t-SNE to reduce the dimensionality of the features to 2D
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(intermediate_representations)

# Visualize the reduced features in a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.5)
plt.title("t-SNE Visualization of Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.savefig("reports/figures/tsne_visualization.png")  # Save the visualization to a file
plt.show()
