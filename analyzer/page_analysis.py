import os
import PyPDF2
import numpy as np
from PyPDF2.errors import PdfReadError

def count_pages(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)
    except (PdfReadError, KeyError) as e:
        print(f"Error reading {pdf_path}: {e}")
        return np.nan  # Return NaN in case of an error

def metrics_annotated_dataset(pdf_folder_path, html_folder_path):
    print("Calculating metrics for annotated dataset...")
    page_counts = []
    filenames = []
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith('.pdf'):
            # Check if corresponding HTML file exists
            html_filename = filename.replace('.pdf', '.html')
            html_path = os.path.join(html_folder_path, html_filename)
            if os.path.exists(html_path):
                pdf_path = os.path.join(pdf_folder_path, filename)
                page_count = count_pages(pdf_path)
                page_counts.append(page_count)
                filenames.append(filename)

    page_counts = np.array(page_counts)  # Convert to NumPy array for easier handling

    if not np.any(~np.isnan(page_counts)):
        print("No valid PDF files with corresponding HTML files found.")
        return

    # Finding largest and smallest files, ignoring NaNs
    valid_indices = ~np.isnan(page_counts)
    largest_file_idx = np.nanargmax(page_counts)
    smallest_file_idx = np.nanargmin(page_counts)

    # Calculating average number of pages, ignoring NaNs
    average_pages = np.nanmean(page_counts)

    # Calculating quartiles, ignoring NaNs
    quartiles = np.nanpercentile(page_counts[valid_indices], [25, 50, 75])

    print(f"Largest file: {filenames[largest_file_idx]} with {page_counts[largest_file_idx]} pages")
    print(f"Smallest file: {filenames[smallest_file_idx]} with {page_counts[smallest_file_idx]} pages")
    print(f"Average number of pages: {average_pages}")
    print(f"Quartiles of number of pages: 25%={quartiles[0]}, 50% (median)={quartiles[1]}, 75%={quartiles[2]}")


def metrics_full_dataset(folder_path):
    print("Calculating metrics for full dataset...")
    page_counts = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            page_count = count_pages(file_path)
            page_counts.append(page_count)
            filenames.append(filename)

    page_counts = np.array(page_counts)  # Convert to NumPy array for easier handling

    if not np.any(~np.isnan(page_counts)):
        print("No valid PDF files found in the folder.")
        return

    # Finding largest and smallest files, ignoring NaNs
    valid_indices = ~np.isnan(page_counts)
    largest_file_idx = np.nanargmax(page_counts)
    smallest_file_idx = np.nanargmin(page_counts)

    # Calculating average number of pages, ignoring NaNs
    average_pages = np.nanmean(page_counts)

    # Calculating quartiles, ignoring NaNs
    quartiles = np.nanpercentile(page_counts[valid_indices], [25, 50, 75])

    print(f"Largest file: {filenames[largest_file_idx]} with {page_counts[largest_file_idx]} pages")
    print(f"Smallest file: {filenames[smallest_file_idx]} with {page_counts[smallest_file_idx]} pages")
    print(f"Average number of pages: {average_pages}")
    print(f"Quartiles of number of pages: 25%={quartiles[0]}, 50% (median)={quartiles[1]}, 75%={quartiles[2]}")


pdf_folder_path = '/home/fuchs/Desktop/dodis/dodo/docs_p1/pdf'
html_folder_path = '/home/fuchs/Desktop/dodis/dodo/docs_p1/html'
# metrics full dataset
metrics_full_dataset(pdf_folder_path)
# metrics annotated dataset
metrics_annotated_dataset(pdf_folder_path, html_folder_path)
