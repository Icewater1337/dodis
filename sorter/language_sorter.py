import json
import os
import shutil
from pathlib import Path

from langdetect import detect


def determine_language(text):
    try:
        lang = detect(text)
        if lang in ['de', 'fr', 'it', 'en']:
            return lang
    except:
        pass
    return None


def main(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(source_folder, filename)
            json_filepath = os.path.join(source_folder.replace("tessdata","json"), filename.replace('.txt', '.json'))
            pdf_filepath = os.path.join(source_folder.replace("tessdata","pdf"), filename.replace('.txt', '.pdf'))

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            lang = determine_language(content)

            with open(json_filepath, 'r') as f:
                data = json.load(f)
                json_doc_lang = data['data']['langCode']

            if json_doc_lang != lang:
                print(f"Language mismatch for {filename}: {json_doc_lang} vs {lang}, skipping file")
                # continue

                # assert json_doc_lang == lang, f"Language mismatch for {filename}: {json_doc_lang} vs {lang}"


            if lang:
                pdf_subfolder = os.path.join(destination_folder, f"{lang}/pdf")
                txt_subfolder = os.path.join(destination_folder, f"{lang}/txt")
                Path(pdf_subfolder).mkdir(parents=True, exist_ok=True)
                Path(txt_subfolder).mkdir(parents=True, exist_ok=True)

                try:
                    shutil.copy(pdf_filepath, pdf_subfolder)
                except FileNotFoundError:
                    print(f"Could not find file {pdf_filepath}")
                    continue
                shutil.copy(filepath, txt_subfolder)


if __name__ == "__main__":
    src_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/tessdata/"
    dest_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/"  # Provide the path to your destination folder
    main(src_folder, dest_folder)
