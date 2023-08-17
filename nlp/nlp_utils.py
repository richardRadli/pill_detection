import re


def four_word_label(text):
    splitted_label = []
    for label in text:
        split = label.split(' ')
        first_four_element = split[:3]
        first_four_element_string = ' '.join([str(elem) for elem in first_four_element])
        splitted_label.append(first_four_element_string)
    return splitted_label


def label_cleaning(data, label: str):
    clean_labels = []
    for label in data[label]:
        clean_label = re.sub(r'\W+', ' ', label).strip()
        clean_labels.append(clean_label)
    return clean_labels
