import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tqdm import tqdm

from config.config import ConfigStreamNetwork
from config.config_selector import sub_stream_network_configs
from utils.utils import find_latest_directory, file_reader


def avg_list(list_name):
    return sum(list_name) / len(list_name)


def process_data_avg(txt_file):
    with open(txt_file, 'r') as f:
        data = eval(f.read())

    num_samples = []

    for idx, value in enumerate(data):
        num_samples.append(len(data[idx][0]))

    return avg_list(num_samples)


def process_data(txt_file):
    with open(txt_file, 'r') as f:
        data = eval(f.read())

    num_samples = [len(entry[0]) for entry in data]

    return num_samples


def plot_hard_samples_per_epoch(files, stream_type):
    for i, f in tqdm(enumerate(files), total=len(files), desc="Processing epochs"):
        hard_samples = process_data(f)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=np.arange(len(hard_samples)), y=hard_samples)
        plt.ylabel("Number of Hard Triplets")
        plt.xlabel("Samples")
        plt.title(f"Epoch {i + 1}")
        plt.tight_layout()
        plt.savefig(f"C:/Users/ricsi/Desktop/plots/%s_{i}.jpg" % stream_type, dpi=400)
        plt.close()


def plot_hard_samples_avg(files, stream_type):
    avg_hard_samples = []
    for i, f in enumerate(files):
        hs = process_data_avg(f)
        avg_hard_samples.append(hs)

    ax = sns.barplot(x=np.arange(len(avg_hard_samples)), y=avg_hard_samples)
    ax.bar_label(ax.containers[0])
    plt.ylabel("Number of Average Hard Triplets")
    plt.xlabel("Epoch")
    plt.savefig(f"C:/Users/ricsi/Desktop/plots/%s_avg.jpg" % stream_type, dpi=400)
    plt.close()


def main(stream_type):
    cfg = ConfigStreamNetwork().parse()
    sub_stream_configs = sub_stream_network_configs(cfg)
    path_to_folder = (
        sub_stream_configs.get(stream_type).get("hardest_samples").get(cfg.type_of_net).get(cfg.dataset_type)
    )
    latest_folder = find_latest_directory(path_to_folder)
    files = file_reader(latest_folder, "txt")

    plot_hard_samples_per_epoch(files, stream_type)
    plot_hard_samples_avg(files, stream_type)


if __name__ == '__main__':
    main(stream_type="Texture")
