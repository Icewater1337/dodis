import os
import csv

root_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it"  # replace with the actual path
csv_file = "dodis_docs_it.csv"

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["name", "language", "handwritten", "year_range"])

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.pdf'):
                name = filename
                language = os.path.basename(root_folder)
                handwritten = "handwritten" in foldername
                year_range = None
                for yr in foldername.split(os.sep):
                    if "-" in yr and yr[0].isdigit():  # simple check for year range pattern
                        year_range = yr
                        break
                writer.writerow([name, language, handwritten, year_range])
