import matplotlib.pyplot as plt
import os
import pandas as pd


def get_classes(dir_images, dir_data):
    classes = set()

    for filename in os.listdir(dir_images):
        if filename.endswith('.png'):
            class_name = filename.split('_')[2:-1]
            classes.add('_'.join(class_name))

    for filename in os.listdir(dir_data):
        if filename.endswith('.txt'):
            class_name = filename.split('_')[2:-1]
            classes.add('_'.join(class_name))

    classes = sorted(classes)

    return {class_name: 0 for class_name in classes}


def calculate_proportions(dir_images, class_counts):
    for filename in os.listdir(dir_images):
        if filename.endswith('.png'):
            class_name = '_'.join(filename.split('_')[2:-1])
            class_counts[class_name] += 1

    total_count = len(os.listdir(dir_images))
    proportions = {}

    for class_name, count in class_counts.items():
        proportion = (count / total_count) * 100
        proportions[class_name] = proportion

    return proportions


def plot_data(proportions, class_counts, threshold=0.2):
    df = pd.DataFrame.from_dict(proportions, orient='index', columns=['Proportion'])
    df.index.name = 'Class'
    df['Instances'] = [class_counts[class_name] for class_name in df.index]
    df.sort_values(by=['Proportion'], ascending=False, inplace=True)

    pd.options.display.float_format = '{:.4f}'.format
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df)

    # Calculate threshold value
    mean = df['Proportion'].mean()
    std = df['Proportion'].std()
    threshold_value_above = mean + threshold * std
    threshold_value_below = mean - threshold * std

    plt.figure(figsize=(20, 10))
    bar_colors = ['tab:red' if proportion > threshold_value_above
                  else 'tab:green' if proportion < threshold_value_below
                  else 'tab:blue' for proportion in df['Proportion']]
    plt.bar(df.index, df['Proportion'], color=bar_colors)
    plt.xlabel('Class')
    plt.ylabel('Proportion (%)')
    plt.title('Class Proportions')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def calculate_imbalance_ratio(class_counts):
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    print(f'Imbalance ratio of the dataset is: {imbalance_ratio}')


def main():
    main_dir = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/ogyi/full_img_size/unsplitted"
    images_directory = os.path.join(main_dir, "images")
    labels_directory = os.path.join(main_dir, "labels")

    number_of_classes = get_classes(images_directory, labels_directory)
    proportion_value = calculate_proportions(images_directory, number_of_classes)
    calculate_imbalance_ratio(number_of_classes)
    plot_data(proportion_value, number_of_classes)


if __name__ == "__main__":
    main()
