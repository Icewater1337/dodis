import json
import os
import shutil
import sys
from pathlib import Path

from langdetect import detect
from loguru import logger

def determine_language(text):
    try:
        lang = detect(text)
        if lang in ['de', 'fr', 'it', 'en']:
            return lang
    except:
        pass
    return None


def sort_by_language(source_folder, destination_folder):
    pfd_not_found = 0
    language_not_found = 0
    language_missmatch = 0
    skip_file_no_txt = 0
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    filenames = list(os.listdir(source_folder))
    for filename in filenames:
        if not filename.endswith(".txt"):
            logger.info(f"Skipping file {filename}, not a text file")
            skip_file_no_txt += 1
        else:
            filepath = os.path.join(source_folder, filename)
            json_filepath = os.path.join(source_folder.replace("tessdata","json"), filename.replace('.txt', '.json'))
            pdf_filepath = os.path.join(source_folder.replace("tessdata","pdf"), filename.replace('.txt', '.pdf'))

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            lang = determine_language(content)
            if not lang:
                logger.error(f"Could not determine language for {filename}, skipping file")
                language_not_found += 1
                continue

            with open(json_filepath, 'r') as f:
                data = json.load(f)
                json_doc_lang = data['data']['langCode']


            if json_doc_lang != lang:
                logger.error(f"Language mismatch for {filename}: {json_doc_lang} vs {lang}, skipping file")
                language_missmatch += 1
                continue

            pdf_subfolder = os.path.join(destination_folder, f"{lang}/pdf")
            txt_subfolder = os.path.join(destination_folder, f"{lang}/txt")
            Path(pdf_subfolder).mkdir(parents=True, exist_ok=True)
            Path(txt_subfolder).mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy(pdf_filepath, pdf_subfolder)
                shutil.copy(filepath, txt_subfolder)
            except FileNotFoundError:
                logger.error(f"Could not find file {pdf_filepath}")
                pfd_not_found += 1
                continue
    logger.info(f"Total files: {len(filenames)}")
    logger.info(f"Language not found: {language_not_found}")
    logger.info(f"Language missmatch: {language_missmatch}")
    logger.info(f"PDF not found: {pfd_not_found}")
    logger.info(f"Skipped files: {skip_file_no_txt}")
    logger.info(f"Total: {language_not_found + language_missmatch + pfd_not_found + skip_file_no_txt}")


if __name__ == "__main__":
    src_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/tessdata/"
    dest_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted2/"  #
    log_file = os.path.join(dest_folder, 'language_sorted.log')

    log_level = "TRACE"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add(log_file, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)

    sort_by_language(src_folder, dest_folder)
