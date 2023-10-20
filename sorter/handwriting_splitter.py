import os
import shutil

import pytesseract
from pdf2image import convert_from_path


def pdf_to_images(pdf_path):
    return convert_from_path(pdf_path)

def get_ocr_accuracy(image):
    # Get OCR output
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Calculate the number of recognized words
    num_recognized_words = sum([1 for conf in ocr_result['conf'] if int(conf) != -1])

    # Calculate accuracy as the ratio of recognized words to total words
    if len(ocr_result['text']) == 0:
        return 0
    accuracy = num_recognized_words / len(ocr_result['text'])

    return accuracy

def classify_text(pdf_path, threshold=0.75):
    images = pdf_to_images(pdf_path)
    accuracies = [get_ocr_accuracy(img) for img in images]

    # If average accuracy across pages is greater than the threshold, classify as computer-written
    if sum(accuracies)/len(accuracies) > threshold:
        return "Computer-Written", sum(accuracies)/len(accuracies)
    else:
        return "Handwritten", sum(accuracies)/len(accuracies)


def classify_and_move(base_directory, dest_handwritten, dest_computer):
    for folder in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder, 'pdf')

        if os.path.isdir(folder_path) and any(file.endswith('.pdf') for file in os.listdir(folder_path)):
            for pdf_file in os.listdir(folder_path):
                pdf_path = os.path.join(folder_path, pdf_file)

                classification, avg_acc = classify_text(pdf_path)

                if classification == "Handwritten":
                    dest_folder = os.path.join(dest_handwritten, folder, 'pdf')

                else:
                    dest_folder = os.path.join(dest_computer, folder, 'pdf')

                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

                print(f"move {classification} file {pdf_file} with avg acc {avg_acc} to {dest_folder}")
                shutil.copy(pdf_path, os.path.join(dest_folder, pdf_file))

# pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs/fr/sorted/handwritten/1849/pdf/41027.pdf"

base_directory = '/home/fuchs/Desktop/dodis/dodo/docs/de/sorted/'  # e.g., '/home/user/documents'
# base_directory = '/home/fuchs/Desktop/dodis/dodo/docs/tesdt/'  # e.g., '/home/user/documents'
# dest_handwritten = '/home/fuchs/Desktop/dodis/dodo/docs/tesdt/handwritten'
dest_handwritten = '/home/fuchs/Desktop/dodis/dodo/docs/de/sorted/handwritten'
# dest_computer = '/home/fuchs/Desktop/dodis/dodo/docs/tesdt/computer'
dest_computer = '/home/fuchs/Desktop/dodis/dodo/docs/de/sorted/computer'

classify_and_move(base_directory, dest_handwritten, dest_computer)

# print(classify_text(pdf_path))
