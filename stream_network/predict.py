import numpy as np
import torch
from stream_network import StreamNetwork
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from dataset_loader_predict import TestDataset
from torch.utils.data import DataLoader


def extract_features(model, dataloader):
    features = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            images, lbls = data
            outputs = model(images)
            features.append(outputs)
            labels.append(lbls)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


dataset_dir = "E:/users/ricsi/IVM/images/bounding_box/"
type_of_network = "RGB" # or "Contour" or "Texture"

# create the dataset and dataloader
dataset = TestDataset(dataset_dir, type_of_network)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

list_of_channels = [3, 64, 96, 128, 256, 384, 512]

model = StreamNetwork(list_of_channels)
state_dict = torch.load('E:/users/ricsi/IVM/data/stream_rgb_model_weights/epoch_9.pt')
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

test_features, test_labels = extract_features(model, dataloader)

distances_c = cosine_similarity(test_features)

nearest_neighbors = np.argsort(-distances_c, axis=1)[:, :3]

# Assign class label based on majority vote
labels = np.array(test_labels)
predicted_labels = []
for neighbors in nearest_neighbors:
    neighbor_labels = labels[neighbors]
    counts = np.bincount(neighbor_labels)
    predicted_label = np.argmax(counts)
    predicted_labels.append(predicted_label)

print(np.shape(predicted_labels))