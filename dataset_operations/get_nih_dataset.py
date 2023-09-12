import csv
import logging
import os
import re
import requests
import xml.etree.ElementTree as ElementTree

from tqdm import tqdm

from utils.utils import setup_logger


class NIHDatasetDownloader:
    def __init__(self):
        setup_logger()

        self.dataset_info_path = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/directory_of_images.txt"
        self.root_url = "https://data.lhncbc.nlm.nih.gov/public/Pills/"

    @staticmethod
    def path_selector():
        return {
            "ref": {
                "output_file": "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/ref.txt",
                "download_directory": "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/ref",
                "feature_csv_file_name": "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/ref_output.csv"
            },
            "query": {
                "output_file": "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/query.txt",
                "download_directory": "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/query",
                "feature_csv_file_name": "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/query_output.csv"
            }
        }

    def get_image_data(self):
        ref_list = []
        query_list = []

        with open(self.dataset_info_path, 'r') as file:
            for line in file:
                data = line.strip().split('|')
                if len(data) >= 4:
                    sublist = [data[0], data[2], data[3], data[4]]
                    if data[3] == "MC_CHALLENGE_V1.0":
                        ref_list.append(sublist)
                    elif data[3] == "C3PI_Test":
                        query_list.append(sublist)

        return ref_list, query_list

    def write_lines_to_file(self, lines: list, operation: str):
        with open(self.path_selector().get(operation).get("output_file"), 'w') as file:
            for line in lines:
                file.write('|'.join(line) + '\n')

    def count_unique_class_labels(self, operation):
        unique_class_labels = set()

        # Open the file for reading
        with open(self.path_selector().get(operation).get("output_file"), 'r') as file:
            for line in file:
                parts = line.strip().split('|')
                if len(parts) >= 4:
                    class_label = parts[-1]
                    unique_class_labels.add(class_label)

        return len(unique_class_labels), unique_class_labels

    @staticmethod
    def display_statistics(unique_labels_ref, unique_labels_query):
        common_labels = unique_labels_ref.intersection(unique_labels_query)
        logging.info(f"Common class labels between ref and query: {common_labels}, "
                     f"\nnumber of common classes: {len(common_labels)}")

        unique_labels_ref_only = unique_labels_ref - unique_labels_query
        logging.info(f"Class labels unique to ref: {unique_labels_ref_only}, "
                     f"\nnumber of unique ref classes: {len(unique_labels_ref_only)}")
        #
        unique_labels_query_only = unique_labels_query - unique_labels_ref
        logging.info(f"Class labels unique to query: {unique_labels_query_only}, "
                     f"\nnumber of unique query classes: {len(unique_labels_query_only)}")

    @staticmethod
    def convert_element(element):
        return re.sub(r'\s+|-|/|\.', '_', element)

    def download_images(self, list_to_process, operation):
        """

        :param list_to_process:
        :param operation:
        :return:
        """

        download_directory = self.path_selector().get(operation).get("download_directory")
        if not os.path.exists(download_directory):
            os.makedirs(download_directory)

        for item in tqdm(list_to_process):
            image_url = self.root_url + item[1]
            response = requests.get(image_url)

            if response.status_code == 200:
                image_filename = os.path.basename(item[1])

                if image_filename.endswith("WMV"):
                    logging.warning(f"WMV file detected, skipped: {image_filename}")
                    continue

                pill_name = item[3]
                pill_directory = os.path.join(download_directory, pill_name)

                class_name = os.path.basename(pill_directory)
                class_name = self.convert_element(class_name)

                pill_directory = pill_directory.split("\\")[0]
                pill_directory = os.path.join(pill_directory, class_name)

                if not os.path.exists(pill_directory):
                    os.makedirs(pill_directory)

                full_path = os.path.join(pill_directory, image_filename)
                with open(full_path, 'wb') as file:
                    file.write(response.content)
                logging.info(f"Downloaded: {image_filename} to {pill_name}")
            else:
                logging.error(f"Failed to download: {image_url}")

    @staticmethod
    def process_xml(xml_url, given_ndc11):
        response = requests.get(xml_url)

        if response.status_code == 200:
            xml_data = response.content
            root = ElementTree.fromstring(xml_data)

            for image_elem in root.findall(".//Image"):
                ndc11 = image_elem.find("NDC11").text
                if ndc11 == given_ndc11:
                    pill_name = image_elem.find("ProprietaryName").text
                    imprint = image_elem.find("Imprint").text
                    imprinttype = image_elem.find("ImprintType").text
                    color = image_elem.find("Color").text
                    shape = image_elem.find("Shape").text
                    score = image_elem.find("Score").text
                    symbol = image_elem.find("Symbol").text
                    size_of_pill = image_elem.find("Size").text
                    return pill_name, imprint, imprinttype, color, shape, score, symbol, size_of_pill
            return None, None, None, None, None, None, None, None
        else:
            logging.error(f"Failed to download XML: {xml_url}")
            return None, None, None, None, None, None, None, None

    def write_attributes_to_csv(self, list_to_process, operation):
        printed_values = set()

        with open(self.path_selector().get(operation).get("feature_csv_file_name"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=",")

            for item in tqdm(list_to_process, desc="Acquiring features from the xml files"):
                directory_name = item[1].split("/")[0]
                if item[0] not in printed_values:
                    printed_values.add(item[0])
                    pill_name, imprint, imprinttype, color, shape, score, symbol, size_of_pill = self.process_xml(
                        "https://data.lhncbc.nlm.nih.gov/public/Pills/ALLXML/%s.xml" % directory_name, item[0])
                    imprint_parts = imprint.split(';') if imprint else []
                    imprint_parts = '_'.join(imprint_parts)
                    imprint_parts_list = [imprint_parts]
                    writer.writerow(
                        [pill_name] + imprint_parts_list + [imprinttype, color, shape, score, symbol, size_of_pill])

    def main(self):
        ref_list, query_list = self.get_image_data()

        self.write_lines_to_file(ref_list, operation="ref")
        self.write_lines_to_file(query_list, operation="query")

        count_ref, unique_labels_ref = self.count_unique_class_labels(operation="ref")
        logging.info(f"Number of unique ref class labels: {count_ref}")

        count_query, unique_labels_query = self.count_unique_class_labels(operation="query")
        logging.info(f"Number of unique query class labels: {count_query}")

        self.display_statistics(unique_labels_ref, unique_labels_query)

        # self.download_images(ref_list, "ref")
        # self.write_attributes_to_csv(ref_list, operation="ref")

        self.download_images(query_list, "query")
        self.write_attributes_to_csv(query_list, operation="query")


if __name__ == "__main__":
    try:
        nih_dataset = NIHDatasetDownloader()
        nih_dataset.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
