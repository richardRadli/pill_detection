import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from sklearn.manifold import TSNE

from nlp_utils import label_cleaning, four_word_label


def clean_text(text):
    regex_for_data_cleaning = re.compile('[()\u201E“”"\u2019\u0027{}\\[\\]|@,;]')
    text = regex_for_data_cleaning.sub('', str(text))
    return text


def plot_one_column_with_count(data, name):
    plt.figure(figsize=(12, 10))
    counts = data[name].value_counts()

    if '' in counts:
        counts = counts.rename({'': 'n.a.'})

    column_colors = ['#FFC300', '#FF5733', '#903F0C', '#900C3F', '#966478', '#082A61', '#0C903F', '#5E12D7', '#0ED836']
    counts.plot.bar(color=column_colors)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.14, top=0.9, wspace=0.3, hspace=0.3)
    plt.title(name, fontweight='bold')
    plt.ylabel('Count (unit)',  fontweight='bold', fontsize=12)
    plt.xlabel('Groups', fontweight='bold', fontsize=12)

    for j, v in enumerate(counts):
        plt.text(j, v + 0.4, str(v), color='black', ha='center', fontweight='bold', fontsize=12)

    plt.show()


def count_the_max_element_of_a_column(data, name_of_the_column):
    max_element = data[name_of_the_column].iat[0]
    for column in data[name_of_the_column]:
        if column > max_element:
            max_element = column
    return max_element


def create_color_subcolumn(data):
    list_of_colors_subcolumn = []
    for color in range(1, count_the_max_element_of_a_column(data, 'Színek száma') + 1):
        list_of_colors_subcolumn.append('Szín_' + str(color))
    return list_of_colors_subcolumn


def create_imprint_subcolumn(data):
    list_of_imprints_subcolumn = []
    for imprint in range(1, count_the_max_element_of_a_column(data, 'Feliratok száma') + 1):
        list_of_imprints_subcolumn.append('Felirat_' + str(imprint))
    return list_of_imprints_subcolumn


def concatenate_list_color_and_shape(data):
    concatenated_list = []
    for index, row in data.iterrows():
        concatenated_list.append(' '.join(row[['Szín_1', 'Alak']].astype(str)))
    return concatenated_list


def visualization(new_values, list_of_labels):
    x = new_values[:, 0] + np.random.normal(0, 100, len(new_values))  # add jittering to x values
    y = new_values[:, 1] + np.random.normal(0, 100, len(new_values))  # add jittering to y values

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(list_of_labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points',
                     ha='right', va='bottom')
    plt.show()


def main():
    root_directory = os.getcwd()
    data = pd.read_csv(os.path.join(root_directory, 'splitted_csv.csv', ), encoding='utf-8')

    data.dropna(how="any", inplace=True, axis=1)
    data.columns = ['Név', 'Gyógyszerforma', 'Szín', 'Színek száma', 'Alak', 'Alakok száma', 'Domborúság',
                    'Él', 'Törővonal', 'Felirat', 'Feliratok száma']
    # egy színek száma oszlop behozása int típussal
    data = data.astype({'Színek száma': 'int'})
    data = data.astype({'Feliratok száma': 'int'})
    data = data.astype({'Alakok száma': 'int'})

    list_of_colors_subcolumn = create_color_subcolumn(data)

    data[list_of_colors_subcolumn] = data['Szín'].str.split(',', expand=True)

    list_of_imprints_subcolumn = create_imprint_subcolumn(data)

    data[list_of_imprints_subcolumn] = data['Felirat'].str.split(',', expand=True)

    data['Törővonal'] = data['Törővonal'].replace({True: '1', False: '0'})

    data.fillna("", inplace=True)


    data[:] = np.vectorize(clean_text)(data)
    # data['Gyógyszerforma'] = data.Gyógyszerforma.apply(clean_text)
    # data['Szín'] = data.Szín.apply(clean_text)
    # data['Szín_1'] = data.Szín_1.apply(clean_text)
    # data['Szín_2'] = data.Szín_2.apply(clean_text)
    # data['Szín_3'] = data.Szín_3.apply(clean_text)
    # data['Szín_4'] = data.Szín_4.apply(clean_text)
    # data['Szín_5'] = data.Szín_5.apply(clean_text)
    # data['Alak'] = data.Alak.apply(clean_text)
    # data['Alak_1'] = data.Alak_1.apply(clean_text)
    # data['Alak_2'] = data.Alak_2.apply(clean_text)
    # data['Domborúság'] = data.Domborúság.apply(clean_text)
    # data['Él'] = data.Él.apply(clean_text)
    # data['Törővonal'] =data.Törővonal.apply(clean_text)
    # data['Felirat'] = data.Felirat.apply(clean_text)
    # data['Felirat_1'] = data.Felirat_1.apply(clean_text)
    # data['Felirat_2'] = data.Felirat_2.apply(clean_text)

    # data = data[['Gyógyszerforma', 'Szín', 'Szín_1', 'Szín_2', 'Szín_3', 'Alak', 'Alak_1', 'Alak_2', 'Domborúság',
    # 'Él', 'Törővonal', 'Felirat', 'Felirat_1', 'Felirat_2']].apply(clean_text)


    # Statisztika és függvények kirajzolása

    # # print(data.groupby('Szín_1').describe())
    plot_one_column_with_count(data, 'Szín_1')
    # # print(data.groupby('Alak').describe())
    # plot_one_column_with_count('Alak')
    # # print(data.groupby('Alak_1').describe())
    # # plot_one_column_with_count('Alak_1')
    # # print(data.groupby('Törővonal').describe())
    # plot_one_column_with_count('Törővonal')
    # # print(data.groupby('Domborúság').describe())
    # plot_one_column_with_count('Domborúság')

    # Azokat az oszlopokat, amiket felosztottunk további oszlopokra, eldobjuk
    # data = data.drop(columns=['Szín', 'Alak', 'Felirat'])
    data = data.drop(columns=['Szín', 'Felirat'])

    # Label Encoding alkalmazása az alábbi oszlopokra

    # le = preprocessing.LabelEncoder()
    # # Dinamikusan kezeljük a listákat
    # list_of_properties = ['Gyógyszerforma', 'Alak', 'Domborúság', 'Él', 'Törővonal']
    # # a gyógyszerforma után hozzáadjuk a színek felosztott listáját
    # list_of_properties[1:1] = list_of_colors_subcolumn
    # list_of_properties.extend(list_of_imprints_subcolumn)
    # print(str(list_of_properties))
    # # majd erre a frissített listára alkalmazzuk a label encodingot
    # data = data[list_of_properties].apply(le.fit_transform)

    # print(data.to_string())

    # Csak a szín és az alak alapján történő abrázolás

    # összevonjuk az elsődlges szín és az alak oszlopait egy listába
    concatenated_list = concatenate_list_color_and_shape(data)

    primary_color_list = data['Szín_1'].to_list()
    primary_shape_list = data['Alak'].to_list()

    print('Primary color list: ')
    print(primary_color_list)

    print('Primary shape list: ')
    print(primary_shape_list)

    # a lista alapján készítünk egy új oszlopot a df-be
    data['concatenated'] = pd.Series(concatenated_list)
    plot_one_column_with_count(data, 'concatenated')
    print(concatenated_list)

    # a labelek megtisztítása a \t-ktől és a mondat eleji spaceek eltávolítása
    clean_labels = label_cleaning(data)

    # létrehozunk egy új oszlopot a tisztított labeleknek
    data['clean_label'] = clean_labels

    # ezt az új oszlopot használjuk a 4 szavas labelek elkészítéséhez
    list_of_labels = four_word_label(data['clean_label'])

    # betöltjük a magyar Spacy modellt
    nlp = spacy.load("hu_core_news_lg")

    # az összevont tulajdonságokat tartalmazó listából vektorokat képzünk a HuSpacy modellel
    vectors = np.array([nlp(sentence).vector for sentence in primary_color_list])

    # a k-klaszterezés (lehet, hogy nem kell)

    # a reprodukálás érdekében a random seed statikussá állítása
    random_seed = 23
    np.random.seed(random_seed)

    # k-klaszterezési modell illesztése a vektorokhoz
    # kmeans_model = KMeans(n_clusters=num_clusters, random_state=random_seed)
    # kmeans_model.fit(vectors)

    # t-SNE alapú dimenzió redukció
    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=5000, random_state=random_seed)
    new_values = tsne_model.fit_transform(vectors)

    visualization(new_values, list_of_labels)


if __name__ == '__main__':
    main()
