import logging
import numpy as np
import os
import pandas as pd
import spacy

from scipy.spatial.distance import pdist, squareform
from collections import OrderedDict
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

from config.const import NLP_DATA_PATH
from nlp_utils import four_word_label, label_cleaning
from utils.utils import create_timestamp, measure_execution_time, setup_logger


class TextNLPAnalysis:
    def __init__(self):
        self.timestamp = create_timestamp()

    @staticmethod
    def load_data():
        directory_path = NLP_DATA_PATH.get_data_path("full_sentence_csv")
        files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
        latest_file = max(files, key=os.path.getmtime)

        return (pd.read_csv(latest_file,
                            encoding='utf-8',
                            names=['label', 'text'],
                            on_bad_lines="warn"))

    @staticmethod
    def split_labels_to_words(list_of_labels):
        stopwords = []
        for label in list_of_labels:
            words = label.split()
            stopwords.extend(words)
        return stopwords

    @measure_execution_time
    def sentence_cleaning(self, data, custom_stopwords, custom_stopwords_2, nlp):
        clean_sentences = []
        for sentence in data['text']:
            sentence = sentence.strip()
            doc = nlp(sentence)
            cleaned_sentence = self.token_cleaning(doc, custom_stopwords, custom_stopwords_2)
            clean_sentences.append(cleaned_sentence)
        return clean_sentences

    @staticmethod
    def token_cleaning(doc, custom_stopwords, custom_stopwords_2):
        clean_tokens = []
        for i, token in enumerate(doc):
            if i < 5:
                if token.text.lower() in custom_stopwords:
                    continue
            if not token.is_stop and not token.is_punct and token.lemma_ not in custom_stopwords_2:
                clean_tokens.append(token.lemma_)
        cleaned_sentence = ' '.join(clean_tokens)
        return cleaned_sentence

    @staticmethod
    def calculate_similarity(clean_sentences, cleaned_example):
        scores = []
        nlp = spacy.load("hu_core_news_lg")
        for sentence in clean_sentences:
            doc = nlp(sentence)
            similarity = doc.similarity(nlp(cleaned_example))
            scores.append(similarity)
        return scores

    @measure_execution_time
    def vectorization(self, clean_sentences, nlp):
        return np.array([nlp(sentence).vector for sentence in clean_sentences])

    @staticmethod
    def create_matrix(list_of_labels, matrix):
        dict_words = {}

        for i, matrix_values in enumerate(matrix):
            dict_words[list_of_labels[i]] = matrix_values

        sorted_dict = OrderedDict(sorted(dict_words.items()))
        sorted_matrix = list(sorted_dict.values())
        labels = list(sorted_dict.keys())

        pairwise_distances = pdist(sorted_matrix, metric='euclidean')
        distance_matrix = squareform(pairwise_distances)

        df = pd.DataFrame(distance_matrix)
        df.columns = labels
        df.index = labels
        wb = Workbook()
        ws = wb.active

        for r in dataframe_to_rows(df, header=True, index=True):
            ws.append(r)
            ws.delete_rows(2)
            ws.freeze_panes = ws["B2"]

    def main(self):
        setup_logger()

        try:
            nlp = spacy.load("hu_core_news_lg")
            logging.info("huspacy is already downloaded.")
        except OSError:
            logging.info("huspacy is not downloaded. Downloading...")
            spacy.cli.download('huspacy')
            nlp = spacy.load("hu_core_news_lg")

        data = self.load_data()

        data['clean_label'] = label_cleaning(data, 'label')
        list_of_labels = four_word_label(data['clean_label'])

        custom_stopwords = ['küllem']
        custom_stopwords_2 = ['oldal', 'mindkét']
        custom_stopwords.extend(self.split_labels_to_words(list_of_labels))
        clean_sentences = self.sentence_cleaning(data, custom_stopwords, custom_stopwords_2, nlp)

        vectors = self.vectorization(clean_sentences, nlp)

        word_vector_save_path = os.path.join(NLP_DATA_PATH.get_data_path("nlp_vector"), self.timestamp)
        os.makedirs(word_vector_save_path, exist_ok=True)

        np.save(os.path.join(word_vector_save_path, "word_vectors.npy"),
                {'vectors': vectors, 'clean_sentences': clean_sentences,
                 'list_of_labels': list_of_labels, 'data': data})


if __name__ == '__main__':
    text_nlp = TextNLPAnalysis()
    text_nlp.main()
