import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import spacy

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from config.const import NLP_DATA_PATH
from utils.utils import create_timestamp, measure_execution_time, setup_logger


class TextNLPAnalysis:
    def __init__(self):
        timestamp = create_timestamp()
        self.word_vector_save_path = os.path.join(NLP_DATA_PATH.get_data_path("nlp_vector"), timestamp)
        os.makedirs(self.word_vector_save_path, exist_ok=True)
        self.word_vector_plot_path = os.path.join(NLP_DATA_PATH.get_data_path("word_vector_vis"), timestamp)
        os.makedirs(self.word_vector_plot_path, exist_ok=True)

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
    def four_word_label(text):
        splitted_label = []
        for label in text:
            split = label.split(' ')
            first_four_element = split[:3]
            first_four_element_string = ' '.join([str(elem) for elem in first_four_element])
            splitted_label.append(first_four_element_string)
        return splitted_label

    @staticmethod
    def label_cleaning(data):
        clean_labels = []
        for label in data['label']:
            clean_label = re.sub(r'\W+', ' ', label).strip()
            clean_labels.append(clean_label)
        return clean_labels

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

    @measure_execution_time
    def vectorization(self, clean_sentences, nlp):
        vectors = np.array([nlp(sentence).vector for sentence in clean_sentences])
        np.save(os.path.join(self.word_vector_save_path, "word_vectors.npy"), vectors)
        return vectors

    def visualization_example(self, nlp, custom_stopwords, custom_stopwords_2, clean_sentences, list_of_labels, data, vectors,
                              num_clusters: int = 3, random_seed: int = 23):
        example = 'ovális, világoskék tabletta'
        example = example.strip()
        doc = nlp(example)

        cleaned_example = self.token_cleaning(doc, custom_stopwords, custom_stopwords_2)

        scores = self.calculate_similarity(clean_sentences, cleaned_example)
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

        data['clean_label'] = self.label_cleaning(data)
        list_of_labels = self.four_word_label(data['clean_label'])

        custom_stopwords = ['küllem']
        custom_stopwords_2 = ['oldal', 'mindkét']
        custom_stopwords.extend(self.split_labels_to_words(list_of_labels))
        clean_sentences = self.sentence_cleaning(data, custom_stopwords, custom_stopwords_2, nlp)

        vectors = self.vectorization(clean_sentences, nlp)

        self.visualization_example(nlp, custom_stopwords, custom_stopwords_2, clean_sentences, list_of_labels, data, vectors)


if __name__ == '__main__':
    text_nlp = TextNLPAnalysis()
    text_nlp.main()
