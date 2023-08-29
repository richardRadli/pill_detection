import csv
import logging
import os
import requests
import xml.etree.ElementTree as ET

from tqdm import tqdm

from utils.utils import setup_logger


def get_image_data():
    filename = "C:/Users/ricsi/Documents/project/storage/IVM/datasets/nih/directory_of_images.txt"

    ref_list = []
    test_list = []

    with open(filename, 'r') as file:
        for line in file:
            data = line.strip().split('|')
            if len(data) >= 4:
                sublist = [data[0], data[2], data[3], data[4]]
                if data[3] == "C3PI_Reference":
                    ref_list.append(sublist)
                elif data[3] == "C3PI_Test":
                    test_list.append(sublist)

    return ref_list, test_list


def is_directory_empty(directory):
    return not any(os.scandir(directory))


def download_images(ref_list, download_directory, root_url):
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)

    for item in tqdm(ref_list):
        image_url = root_url + item[1]
        response = requests.get(image_url)

        if response.status_code == 200:
            image_filename = os.path.basename(item[1])
            pill_name = item[3]
            pill_directory = os.path.join(download_directory, pill_name)

            if not os.path.exists(pill_directory):
                os.makedirs(pill_directory)

            if is_directory_empty(pill_directory):
                full_path = os.path.join(pill_directory, image_filename)

                with open(full_path, 'wb') as file:
                    file.write(response.content)
                logging.info(f"Downloaded: {image_filename} to {pill_name}")
            else:
                logging.warning(f"Skipped download: {image_filename} (Directory not empty)")

        else:
            logging.error(f"Failed to download: {image_url}")


def process_xml(xml_url, given_ndc11):
    response = requests.get(xml_url)

    if response.status_code == 200:
        xml_data = response.content
        root = ET.fromstring(xml_data)

        for image_elem in root.findall(".//Image"):
            ndc11 = image_elem.find("NDC11").text
            if ndc11 == given_ndc11:
                imprint = image_elem.find("Imprint").text
                imprinttype = image_elem.find("ImprintType").text
                color = image_elem.find("Color").text
                shape = image_elem.find("Shape").text
                score = image_elem.find("Score").text
                symbol = image_elem.find("Symbol").text
                size_of_pill = image_elem.find("Size").text
                return imprint, imprinttype, color, shape, score, symbol, size_of_pill
        return None, None, None, None, None, None, None
    else:
        logging.error(f"Failed to download XML: {xml_url}")
        return None, None, None, None, None, None, None


def main():
    setup_logger()

    ref_list, test_list = get_image_data()
    # root_url = "https://data.lhncbc.nlm.nih.gov/public/Pills/"
    #
    # download_directory_ref = "C:/Users/ricsi/Desktop/nih/ref"
    # download_images(test_list, download_directory_ref, root_url)
    #
    # download_directory_test = "C:/Users/ricsi/Desktop/nih/test"
    # download_images(test_list, download_directory_test, root_url)
    #

    printed_values = set()

    with open('C:/Users/ricsi/Desktop/nih/ref/output.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")

        for item in ref_list:
            directory_name = item[1].split("/")[0]
            if item[0] not in printed_values:
                printed_values.add(item[0])
                imprint, imprinttype, color, shape, score, symbol, size_of_pill = process_xml(
                    "https://data.lhncbc.nlm.nih.gov/public/Pills/ALLXML/%s.xml" % directory_name, item[0])
                imprint_parts = imprint.split(';')
                imprint_parts = '_'.join(imprint_parts)
                imprint_parts_list = [imprint_parts]
                writer.writerow(imprint_parts_list + [imprinttype, color, shape, score, symbol, size_of_pill])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
