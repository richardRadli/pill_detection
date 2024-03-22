import json
import numpy as np
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
        original_vector = None

        for class_label in self.classes:
            if idx < len(self.data[class_label]):
                original_vector = torch.tensor(self.data[class_label][idx], dtype=torch.float32)
                break
            else:
                idx -= len(self.data[class_label])

        augmented_vector = self.augment(original_vector)

        return original_vector, augmented_vector

    @staticmethod
    def load_json(json_file):
        with open(json_file) as file:
            data = json.load(file)
        return data

    def get_classes(self):
        return self.classes

    def augment(self, feature_vector):
        augmented_lab_values = self.augment_lab_values(feature_vector)
        augmented_fourier_values = self.augment_fourier_values(feature_vector)
        imprint_score_values = self.get_imprint_and_score(feature_vector).tolist()
        augmented_vector = (
            torch.tensor(augmented_lab_values + augmented_fourier_values + imprint_score_values, dtype=torch.float32)
        )
        return augmented_vector

    @staticmethod
    def augment_lab_values(feature_vector):
        lab_value = feature_vector[:12]
        augmented_lab_value = [f + np.random.normal(loc=0, scale=0.1) for f in lab_value]
        return augmented_lab_value

    @staticmethod
    def augment_fourier_values(feature_vector):
        fourier_value = feature_vector[12:49]
        augmented_fourier_value = [f + np.random.normal(loc=0, scale=0.1) for f in fourier_value]
        return augmented_fourier_value

    @staticmethod
    def get_imprint_and_score(feature_vector):
        return feature_vector[49:61]
