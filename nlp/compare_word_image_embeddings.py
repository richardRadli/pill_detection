import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def get_vectors():
    embedded_image_file = torch.load("2023-08-21_14-16-08_ref_vectors.pt")
    embedded_image_vectors = embedded_image_file.get('average')

    embedded_word_vectors = np.load("word_vectors.npy", allow_pickle=True).item()
    embedded_word_keys = embedded_word_vectors.keys()

    word_labels = []
    for key in embedded_word_keys:
        word_labels.append(key)

    embedded_word_vectors = np.stack(list(embedded_word_vectors.values()))

    return embedded_image_vectors, embedded_word_vectors, word_labels


def plot_2d(distances, word_labels, threshold):
    plt.figure(figsize=(12, 8))
    plt.scatter(range(len(distances)), distances, c='blue', marker='o', s=15)
    plt.xlabel('Pill label')
    plt.ylabel('Euclidean Distance')
    plt.title('Euclidean Distances between Image and Word embeddings')

    # Adjust tick labels
    plt.xticks(range(len(word_labels)), word_labels, rotation='vertical', fontsize=8)
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(range(len(word_labels))))

    for x, y in enumerate(distances):
        if y > threshold:
            label = "{:.2f}".format(y)
            plt.annotate(label,
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha="center")

    plt.grid(which="major")
    plt.tight_layout()
    plt.show()


def plot_3d(new_values, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(new_values.shape[0]):
        x, y, z = new_values[i]
        ax.scatter(x, y, z, label=str(i))
        ax.text(x, y, z, str(i), color='black', fontsize=10, ha='right', va='center')

    ax.set_title(title)
    plt.show()


def normalize(your_array):
    min_val = np.min(your_array)
    max_val = np.max(your_array)

    return (your_array - min_val) / (max_val - min_val)


def main():
    embedded_image_vectors, embedded_word_vectors, word_labels = get_vectors()

    # Normalize
    embedded_image_vectors = normalize(embedded_image_vectors)
    embedded_word_vectors = normalize(embedded_word_vectors)

    # Dim. red. and calculate Euclidean distances
    pca_model = PCA(n_components=3, random_state=42)
    new_values_images = pca_model.fit_transform(embedded_image_vectors)
    new_values_words = pca_model.fit_transform(embedded_word_vectors)

    plot_3d(new_values_words, "Word embeddings")
    plot_3d(new_values_images, "Images embeddings")

    distances = np.linalg.norm(new_values_images[:, :2] - new_values_words[:, :2], axis=1)
    plot_2d(distances, word_labels, 0.75)


if __name__ == "__main__":
    main()
