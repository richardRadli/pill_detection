import json
import torch

from torch.utils.data import Dataset


class DataLoaderWordEmbeddingNetwork(Dataset):
    def __init__(self, json_file):
        self.data = self.load_json(json_file)
        self.classes = list(self.data.keys())

    def __len__(self):
        total_samples = sum(len(self.data[class_label]) for class_label in self.classes)
        return total_samples

    def __getitem__(self, idx):
        for class_label in self.classes:
            if idx < len(self.data[class_label]):
                return torch.tensor(self.data[class_label][idx], dtype=torch.float32), class_label
            else:
                idx -= len(self.data[class_label])

    @staticmethod
    def load_json(json_file):
        with open(json_file) as file:
            data = json.load(file)
        return data

    def get_classes(self):
        return self.classes
