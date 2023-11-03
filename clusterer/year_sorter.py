import os
import json
import re
import shutil
from pathlib import Path
import os

# Directories
pdf_dir = '/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it/pdf'
json_dir = '/home/fuchs/Desktop/dodis/dodo/docs_p1/json'
txt_dir = '/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it/txt'
output_dir = '/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it/year_sorted/'




pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
date_pattern = re.compile(r'\[?\.*(\d{1,2})\.*(\d{1,2})\.(\d{4})\.*\]?')
date_pattern_month_only = re.compile(r'\[?\.*(\d{1,2})\.(\d{4})\.*\]?')
date_pattern_year_only = re.compile(r'\[?\.*(\d{4})\.*\]?')

for pdf_file in pdf_files:
    year  = None
    json_file = pdf_file.replace('.pdf', '.json')
    txt_file = pdf_file.replace('.pdf', '.txt')
    print(f"Moving file {json_file}")
    with open(os.path.join(json_dir, json_file), 'r') as f:
        data = json.load(f)
        document_date = data['data']['documentDate']
        match = date_pattern.search(document_date)
        if match:
            day, month, year = match.groups()
        else:
            match = date_pattern_month_only.search(document_date)
            if match:
                month, year = match.groups()
                day = '01'
            else:
                print(f"Could not parse day or month {document_date}, trying year")
                match = date_pattern_year_only.search(document_date)
                if match:
                    year = match.groups()[0]
                    month = '01'
                    day = '01'
        try:
            year = int(year)
        except:
            print(f"Could not parse year {document_date} for document {pdf_file}")
            continue
        # Determine the directory for the year range
        start_year = (year // 5) * 5
        end_year = start_year + 4
        year_range = f"{start_year}-{end_year}"
        # year_range = str(year)
        output_folder_year = os.path.join(output_dir, year_range)
        output_dir_year_pdf = os.path.join(output_folder_year, 'pdf')
        # output_dir_year_txt = os.path.join(output_folder_year, 'txt')

        Path(output_dir_year_pdf).mkdir(parents=True, exist_ok=True)
        # Path(output_dir_year_txt).mkdir(parents=True, exist_ok=True)

        # Move corresponding pdf and json to the new directory

        # os.rename(os.path.join(json_dir, json_file), os.path.join(output_dir_year_json, json_file))
        # try:
        #     shutil.copy(os.path.join(txt_dir, txt_file), os.path.join(output_dir_year_txt, txt_file))
        # except FileNotFoundError:
        #     print(f"Could not find file {txt_file}")
        os.rename(os.path.join(pdf_dir, pdf_file), os.path.join(output_dir_year_pdf, pdf_file))


print("Sorting complete!")



