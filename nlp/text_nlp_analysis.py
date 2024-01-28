import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import spacy

from collections import OrderedDict
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from config.config_selector import nlp_configs
from utils.utils import create_timestamp, measure_execution_time, setup_logger


class TextNLPAnalysis:
    def __init__(self):
        self.timestamp = create_timestamp()
        self.word_vector_save_path = os.path.join(nlp_configs().get("nlp_vector"), self.timestamp)
        os.makedirs(self.word_vector_save_path, exist_ok=True)
        self.num_clusters = 3
        self.random_seed = 23

    @staticmethod
    def load_data():
        """

        :return:
        """

        directory_path = nlp_configs().get("full_sentence_csv")
        files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
        latest_file = max(files, key=os.path.getmtime)

        return (pd.read_csv(latest_file,
                            encoding='utf-8',
                            names=['label', 'text'],
                            on_bad_lines="warn"))

    @staticmethod
    def four_word_label(text):
        """

        :param text:
        :return:
        """

        splitted_label = []
        for label in text:
            split = label.split(' ')
            first_four_element = split[:3]
            first_four_element_string = ' '.join([str(elem) for elem in first_four_element])
            splitted_label.append(first_four_element_string)
        return splitted_label

    @staticmethod
    def label_cleaning(data, label: str):
        """

        :param data:
        :param label:
        :return:
        """

        clean_labels = []
        for label in data[label]:
            clean_label = re.sub(r'\W+', ' ', label).strip()
            clean_labels.append(clean_label)
        return clean_labels

    @staticmethod
    def split_labels_to_words(list_of_labels):
        """

        :param list_of_labels:
        :return:
        """

        stopwords = []
        for label in list_of_labels:
            words = label.split()
            stopwords.extend(words)
        return stopwords

    @measure_execution_time
    def sentence_cleaning(self, data, custom_stopwords, custom_stopwords_2, nlp):
        """

        :param data:
        :param custom_stopwords:
        :param custom_stopwords_2:
        :param nlp:
        :return:
        """

        clean_sentences = []
        for sentence in data['text']:
            sentence = sentence.strip()
            doc = nlp(sentence)
            cleaned_sentence = self.token_cleaning(doc, custom_stopwords, custom_stopwords_2)
            clean_sentences.append(cleaned_sentence)
        return clean_sentences

    @staticmethod
    def token_cleaning(doc, custom_stopwords, custom_stopwords_2):
        """

        :param doc:
        :param custom_stopwords:
        :param custom_stopwords_2:
        :return:
        """

        clean_tokens = []
        for i, token in enumerate(doc):
            if i < 5:
                if token.text.lower() in custom_stopwords:
                    continue
            if not token.is_stop and not token.is_punct and token.lemma_ not in custom_stopwords_2:
                clean_tokens.append(token.lemma_)
        cleaned_sentence = ' '.join(clean_tokens)
        return cleaned_sentence

    @measure_execution_time
    def vectorization(self, clean_sentences, nlp):
        """

        :param clean_sentences:
        :param nlp:
        :return:
        """

        return np.array([nlp(sentence).vector for sentence in clean_sentences])

    def create_matrix(self, list_of_labels, matrix):
        """

        :param list_of_labels:
        :param matrix:
        :return:
        """

        dict_words = {}

        for i, matrix_values in enumerate(matrix):
            dict_words[list_of_labels[i]] = matrix_values

        sorted_dict = OrderedDict(sorted(dict_words.items()))
        sorted_matrix = list(sorted_dict.values())
        labels = list(sorted_dict.keys())

        pairwise_distances = pdist(sorted_matrix, metric='euclidean')
        distance_matrix = squareform(pairwise_distances)

        sns.heatmap(distance_matrix, cmap='viridis', annot=False, fmt=".2f", square=True)
        plt.show()

        df = pd.DataFrame(distance_matrix)
        df.columns = labels
        df.index = labels
        wb = Workbook()
        ws = wb.active

        for r in dataframe_to_rows(df, header=True, index=True):
            ws.append(r)
        ws.delete_rows(2)
        ws.freeze_panes = ws["B2"]

        # Iterate over all columns and adjust their widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except Exception:
                    pass
            adjusted_width = (max_length + 2) * 1.2
            ws.column_dimensions[column_letter].width = adjusted_width

        file_name = os.path.join(nlp_configs().get("vector_distances"), self.timestamp+"_vector_distances.xlsx")
        wb.save(file_name)

    @staticmethod
    def visualization(kmeans_model, new_values, list_of_labels):
        """

        :param kmeans_model:
        :param new_values:
        :param list_of_labels:
        :return:
        """

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
        plt.show()

    def main(self):
        """

        :return:
        """

        setup_logger()

        try:
            nlp = spacy.load("hu_core_news_lg")
        except OSError:
            logging.info("huspacy is not downloaded. Downloading...")
            spacy.cli.download('huspacy')
            nlp = spacy.load("hu_core_news_lg")

        data = self.load_data()

        data['clean_label'] = self.label_cleaning(data, 'label')
        list_of_labels = self.four_word_label(data['clean_label'])

        custom_stopwords = ['küllem']
        custom_stopwords_2 = ['oldal', 'mindkét']
        custom_stopwords.extend(self.split_labels_to_words(list_of_labels))
        clean_sentences = self.sentence_cleaning(data, custom_stopwords, custom_stopwords_2, nlp)

        vectors = self.vectorization(clean_sentences, nlp)

        np.save(os.path.join(self.word_vector_save_path, "word_vectors.npy"),
                {'vectors': vectors,
                 'clean_sentences': clean_sentences,
                 'list_of_labels': list_of_labels,
                 'data': data}
                )

        tsne_model = TSNE(perplexity=25, n_components=2, init='pca', n_iter=5000, random_state=self.random_seed)
        new_values = tsne_model.fit_transform(vectors)
        self.create_matrix(list_of_labels, new_values)

        kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=self.random_seed, n_init=10)
        kmeans_model.fit(vectors)
        self.visualization(kmeans_model, new_values, list_of_labels)


if __name__ == '__main__':
    text_nlp = TextNLPAnalysis()
    text_nlp.main()
