import csv
import docx2txt
import fuzzywuzzy
import fuzzywuzzy.process
import logging
import os
import re
import pandas as pd
import win32com.client

from glob import glob
from fuzzysearch import find_near_matches
from tqdm import tqdm

from config.config_selector import nlp_configs
from utils.utils import create_timestamp, setup_logger


class PillFeatureExtraction:
    def __init__(self):
        setup_logger()
        self.timestamp = create_timestamp()
        self.doc_files = (
            sorted(glob(os.path.join(nlp_configs().get("patient_information_leaflet_doc"),  "*"))))
        self.docx_path = nlp_configs().get("patient_information_leaflet_docx")
        self.dictionary_of_drugs = {}
        self.dictionary_of_drugs_full_sentence = {}
        self.config_lists = self.const_values()

    @staticmethod
    def const_values():
        config = {
            "name_regexes_1":
                ['Információk a felhasználó számára',
                 'Információk a beteg számára',
                 'INFORMÁCIÓK A FELHASZNÁLÓ SZÁMÁRA'],
            "name_regexes_2":
                ['Mielőtt elkezdi szedni ezt a gyógyszert, olvassa',
                 'Mielőtt elkezdi alkalmazni ezt a gyógyszert, olvassa'],
            "name_reference_list":
                ['mg', 'tabletta', 'kapszula'],
            "properties_regexes_1":
                ['külleme és mit tartalmaz a csomagolás'],
            "properties_regexes_2":
                ['a forgalomba hozatali engedély jogosultja', 'Gyártó'],
            "reference_list":
                ['kerek', 'domború', 'jelzés', 'világos', 'színtelen', 'hosszúkás', 'oldal', 'színű', 'korong',
                 'felület', 'csaknem', 'hosszú', 'sárga', 'rózsaszín', 'méretű', 'kék', 'jelölés', 'zöld', 'lila',
                 'elefántcsont', 'alsórészű', 'felsőrészű', 'ovális', 'mélynyomás', 'ellátott', 'méret', 'halvány',
                 'barna'],
            "color_reference_list":
                ['színtelen', 'fehér', 'világoszöld', 'zöld', 'narancssárga', 'halványsárga', 'sötétsárga', 'sárga',
                 'sárgászöld', 'rózsaszín', 'világoskék', 'kék', 'sötétkék', 'piros', 'elefántcsont', 'barna', 'lila',
                 'bordó', 'tejkaramella', 'áttetsző', 'fekete', 'narancsvörös', 'szürke', 'vöröses-barna',
                 'vörösesbarna', 'halványrózsaszín'],
            "shape_reference_list":
                ['kerek', 'ovális', 'korong', 'hosszúkás', 'kapszula', 'háromszög', 'négyszög', 'gömbölyű', 'lencse'],
            "convexity_reference_list":
                ['domború', 'lapos'],
            "edge_reference_list":
                ['metszett'],
            "cut_reference_list":
                ['bemetszés', 'törővonal', 'törés'],
            "sides_difference_reference_list":
                ['egyik', 'másik', 'felső', 'alsó'],
            "sides_same_reference_list":
                ['mindkét'],
            "pharmaceutical_form_reference_list":
                ['tabletta', 'kapszula']
        }

        return config

    @staticmethod
    def get_ref_name_list():
        directory_path = nlp_configs().get("pill_names")
        pill_names_file = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]
        data = pd.read_excel(pill_names_file[0], header=None)
        data = data.applymap(lambda x: x.replace('\xa0', ' ') if isinstance(x, str) else x)
        return [str(cell) for row in data.values for cell in row]

    def convert_doc_files_to_docx(self):
        word = win32com.client.Dispatch("Word.Application")
        word.visible = 0

        for doc in self.doc_files:
            wb = word.Documents.Open(doc)

            file_name, _ = os.path.splitext(os.path.basename(doc))
            out_file = os.path.join(self.docx_path, file_name + ".docx")

            try:
                wb.SaveAs2(out_file, FileFormat=16)
                logging.info(f"{doc} converted to {file_name}.docx")
            except Exception as e:
                logging.info(f"Conversion failed for {doc}: {e}")

            wb.Close()

        word.Quit()

    @staticmethod
    def find_regex_with_fuzzy(reference_list, text, max_l_dist=2):
        matches = []

        for reference in reference_list:
            lower_case_reference = reference.casefold()
            current_matches = find_near_matches(lower_case_reference, text, max_l_dist=max_l_dist)

            if current_matches:
                return current_matches

            matches.extend(current_matches)

        return matches

    @staticmethod
    def edit_name_list(drug_names, name_reference_list):
        name_list = [
            drug_name.replace('\xa0', ' ')
            for drug_name in drug_names.split('\n')
            if drug_name and any(reference in drug_name for reference in name_reference_list)
        ]
        return name_list

    @staticmethod
    def create_new_name_list(edited_name_list, reference_name_list):
        new_name_list = []
        for name in edited_name_list:
            if name in reference_name_list:
                new_name_list.append(name)
        return new_name_list

    @staticmethod
    def extract_with_fuzzy_wuzzy(reference_list, string_to_match, min_ratio=80, limit=None):
        special_strings = ['', '–', '×', '			', '/']

        if string_to_match in special_strings:
            return []

        matches = fuzzywuzzy.process.extract(
            string_to_match, reference_list, limit=limit, scorer=fuzzywuzzy.fuzz.ratio
        )
        close_matches = [match for match, score in matches if score >= min_ratio]

        return close_matches

    @staticmethod
    def clean_list(nested_list):
        return [item for sublist in nested_list for item in sublist if item]

    def set_attribute_list(self, reference_list, splitted_row, min_ratio=None):
        attribute_list = [self.extract_with_fuzzy_wuzzy(reference_list, sub_row, min_ratio) for sub_row in splitted_row]
        attribute_list = self.clean_list(attribute_list)
        return attribute_list

    @staticmethod
    def set_imprint_list(row):
        return re.findall(r'\u201E.*?”|".*?"|“.*?”|\u2019.*?\u2019', row)

    @staticmethod
    def write_list_to_file(lst, name, file):
        file.write(f"{name}: ")

        if not lst:
            file.write('Not in the text')
        else:
            lst_str = ' '.join(lst)
            file.write(lst_str)

        file.write('\n\n')

    def write_attributes_to_file(self, file, color_list, shape_list, convexity_list, edge_list, imprint_list):
        self.write_list_to_file(str(color_list), 'Színek', file)
        self.write_list_to_file(shape_list, 'Alak', file)
        self.write_list_to_file(convexity_list, 'Domborúság', file)
        self.write_list_to_file(edge_list, "Él", file)
        self.write_list_to_file(imprint_list, 'Felirat', file)

    @staticmethod
    def set_and_write_cut_to_file(file, cut_list):
        cut = True if len(cut_list) else False
        file.write('Törővonal/bemetszés: ' + ('van' if cut else 'nincs'))
        file.write('\n\n')
        return cut

    def get_pharmaceutical_form(self, name_list, index):
        splitted_name = name_list[index].split(' ')
        pharmaceutical_form = \
            [self.extract_with_fuzzy_wuzzy(
                self.config_lists.get("pharmaceutical_form_reference_list"), sub_row) for sub_row in splitted_name]
        return self.clean_list(pharmaceutical_form)

    def write_to_csv(self):
        filename = os.path.join(nlp_configs().get("full_sentence_csv"), self.timestamp + "full_sentence_csv_prob.csv")
        with open(filename, "w", encoding='UTF-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self.dictionary_of_drugs_full_sentence.items())

    def main(self):
        reference_name_list = self.get_ref_name_list()
        # self.convert_doc_files_to_docx()
        docx_files = sorted(glob(os.path.join(nlp_configs().get("patient_information_leaflet_docx"), "*")))

        for i, docx in tqdm(enumerate(docx_files), total=len(docx_files), desc="Processing files"):
            text = docx2txt.process(docx)
            lower_case_text = text.casefold()
            end_index_properties_1 = str()
            start_index_properties_2 = str()
            name_list = []

            try:
                name_match_1 = (
                    self.find_regex_with_fuzzy(self.config_lists.get("name_regexes_1"), lower_case_text, max_l_dist=2)
                )
                end_index_name = name_match_1[0].end

                name_match_2 = (
                    self.find_regex_with_fuzzy(self.config_lists.get("name_regexes_2"), lower_case_text, max_l_dist=3)
                )

                start_index_name_2 = name_match_2[0].start
                drug_names = lower_case_text[end_index_name + 1:start_index_name_2:1]

                edited_name_list = self.edit_name_list(drug_names, self.config_lists.get("name_reference_list"))
                name_list = self.create_new_name_list(edited_name_list, reference_name_list)

                properties_match_1 = (
                    self.find_regex_with_fuzzy(
                        self.config_lists.get("properties_regexes_1"), lower_case_text, max_l_dist=2
                    )
                )
                end_index_properties_1 = properties_match_1[0].end

                properties_match_2 = (
                    self.find_regex_with_fuzzy(
                        self.config_lists.get("properties_regexes_2"), lower_case_text, max_l_dist=2
                    )
                )
                start_index_properties_2 = properties_match_2[0].start

            except TypeError:
                logging.error("No regexp in the file!")

            drug_properties = lower_case_text[end_index_properties_1 + 1:start_index_properties_2]
            splitted_properties = drug_properties.split('\n')

            file_basename = os.path.basename(docx)
            filename = file_basename.split(".")[0] + "_results.txt"
            file = open(os.path.join(nlp_configs().get("extracted_features_files"), filename), "w+", encoding="utf-8")
            length_of_name_list = len(name_list)

            index = 0
            for row in splitted_properties:
                if row == '' or row == '\n' or row == '\t':
                    continue
                splitted_row = row.split(' ')

                if any(self.extract_with_fuzzy_wuzzy(self.config_lists.get("reference_list"), sub_row, limit=3) for sub_row in splitted_row):
                    color_list = (
                        self.set_attribute_list(self.config_lists.get("color_reference_list"), splitted_row, 94))
                    shape_list = (
                        self.set_attribute_list(self.config_lists.get("shape_reference_list"), splitted_row, 85))
                    convexity_list = (
                        self.set_attribute_list(self.config_lists.get("convexity_reference_list"), splitted_row, 80))
                    edge_list = (
                        self.set_attribute_list(self.config_lists.get("edge_reference_list"), splitted_row, 80))
                    cut_list = (
                        self.set_attribute_list(self.config_lists.get("cut_reference_list"), splitted_row, 80))
                    imprint_list = self.set_imprint_list(row)

                    if all(len(lst) == 0 for lst in [shape_list, imprint_list, convexity_list]):
                        continue
                    else:
                        self.write_attributes_to_file(
                            file, color_list, shape_list, convexity_list, edge_list, imprint_list)
                        cut = self.set_and_write_cut_to_file(file, cut_list)

                        if length_of_name_list > index:
                            pharmaceutical_form = self.get_pharmaceutical_form(name_list, index)
                            self.write_list_to_file(pharmaceutical_form, 'Gyógyszerforma', file)

                            if len(shape_list) > 1 and len(pharmaceutical_form) == 1:
                                shape_list = [item for item in shape_list if item != 'kapszula']

                            list_of_properties = [pharmaceutical_form, color_list, len(color_list), shape_list,
                                                  len(shape_list), convexity_list,
                                                  edge_list, cut, imprint_list, len(imprint_list)]
                            self.dictionary_of_drugs_full_sentence[name_list[index]] = row
                            self.dictionary_of_drugs[name_list[index]] = list_of_properties
                            index += 1
                            file.write('------------------------------------------------------------------------' + '\n\n')

        self.write_to_csv()


if __name__ == "__main__":
    try:
        pfe = PillFeatureExtraction()
        pfe.main()
    except KeyboardInterrupt as kie:
        logging.error(f'{kie}')
