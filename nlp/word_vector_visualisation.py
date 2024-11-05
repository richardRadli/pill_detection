import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import spacy

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer

from config.nlp_paths_selector import nlp_configs
from text_nlp_analysis import TextNLPAnalysis
from utils.utils import create_timestamp,  setup_logger


class WordVectorVisualisation:
    def __init__(self):
        setup_logger()
        timestamp = create_timestamp()

        self.word_vector_plot_path = os.path.join(nlp_configs().get("word_vector_vis"), timestamp)
        os.makedirs(self.word_vector_plot_path, exist_ok=True)

        self.elbow_path = os.path.join(nlp_configs().get("elbow"), timestamp)
        os.makedirs(self.elbow_path, exist_ok=True)

        self.silhouette_path = os.path.join(nlp_configs().get("silhouette"), timestamp)
        os.makedirs(self.silhouette_path, exist_ok=True)

        self.text_nlp_analysis = TextNLPAnalysis()

    @staticmethod
    def load_vector():
        directory_path = nlp_configs().get("nlp_vector")
        files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
        latest_file = max(files, key=os.path.getmtime)

        return np.load(os.path.join(latest_file, os.listdir(latest_file)[0]), allow_pickle=True)

    def visualization_example(self, clean_sentences, list_of_labels, data, vectors, num_clusters: int = 3,
                              random_seed: int = 23):
        nlp = spacy.load("hu_core_news_lg")
        custom_stopwords = ['küllem']
        custom_stopwords_2 = ['oldal', 'mindkét']
        example = 'ovális, világoskék tabletta'
        example = example.strip()
        doc = nlp(example)

        cleaned_example = self.text_nlp_analysis.token_cleaning(doc, custom_stopwords, custom_stopwords_2)

        scores = self.text_nlp_analysis.calculate_similarity(clean_sentences, cleaned_example)
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        dataframe = self.print_scores(example, indices, list_of_labels, data, scores)
        logging.info(dataframe)

        np.random.seed(random_seed)

        kmeans_model = KMeans(n_clusters=num_clusters,
                              random_state=random_seed,
                              n_init=10)
        kmeans_model.fit(vectors)

        tsne_model = TSNE(perplexity=25,
                          n_components=2,
                          init='pca',
                          n_iter=5000,
                          random_state=random_seed)
        new_values = tsne_model.fit_transform(vectors)

        self.visualization(kmeans_model, new_values, list_of_labels)

    @staticmethod
    def print_scores(example, indices, list_of_labels, data, scores):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        score_data = []
        for i in indices:
            label = list_of_labels[i]
            text = data['text'][i].strip()
            similarity_score = scores[i]
            score_data.append([label, text, similarity_score])

        columns = ['Label', 'Text', 'Similarity Score']
        df = pd.DataFrame(score_data, columns=columns)
        df.insert(0, 'Example Sentence', example)
        return df

    def visualization(self, kmeans_model, new_values, list_of_labels):
        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'gray', 'brown']

        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i], y[i], color=colors[kmeans_model.labels_[i]])
            plt.annotate(list_of_labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right',
                         va='bottom')
        plt.savefig(os.path.join(self.word_vector_plot_path, "word_vectors.png"))

    def visualization_elbow(self, vectors):
        wcss = []
        for i in range(1, 15):
            kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
            kmeans.fit(vectors)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, 15), wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig(os.path.join(self.elbow_path, "elbow.png"))

    def visualization_silhouette(self, vectors):
        num_clusters_range = range(2, 6)
        fig, axs = plt.subplots(len(num_clusters_range), 1, figsize=(10, len(num_clusters_range) * 5))

        for i, num_clusters in enumerate(num_clusters_range):
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            visualizer = SilhouetteVisualizer(kmeans, ax=axs[i])
            visualizer.fit(np.array(vectors))
            axs[i].set_title("num_clusters = {}".format(num_clusters))
            avg_score = visualizer.silhouette_score_
            logging.info("n_clusters = {}, average silhouette score = {}".format(num_clusters, avg_score))
        plt.tight_layout()
        plt.savefig(os.path.join(self.silhouette_path, "silhouette.png"))

    def main(self):
        data = self.load_vector().item()
        vectors = data['vectors']
        clean_sentences = data['clean_sentences']
        list_of_labels = data['list_of_labels']
        data_csv = data['data']

        self.visualization_example(clean_sentences, list_of_labels, data_csv, vectors)
        self.visualization_elbow(vectors)
        self.visualization_silhouette(vectors)


if __name__ == "__main__":
    vis = WordVectorVisualisation()
    vis.main()
