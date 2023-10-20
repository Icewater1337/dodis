from pathlib import Path

import easyocr
import os

import numpy as np
from pdf2image import convert_from_path

def ocr_pdf_folder(input_folder, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['de'])

    # Loop through each PDF in the input folder
    for pdf_file in os.listdir(input_folder):
        if pdf_file.endswith(".pdf"):
            # Convert PDF to images
            images = convert_from_path(os.path.join(input_folder, pdf_file))

            # OCR each image
            full_text = ""
            for image in images:
                image_np = np.array(image)
                result = reader.readtext(image_np)
                for detection in result:
                    full_text += detection[1] + "\n"

            # Save the OCR'd text to the output folder
            output_file_name = os.path.splitext(pdf_file)[0] + ".txt"
            with open(os.path.join(output_folder, output_file_name), 'w', encoding='utf-8') as output_file:
                output_file.write(full_text)

            print(f"Processed {pdf_file} and saved to {output_file_name}")

if __name__ == "__main__":
    INPUT_FOLDER = "input_test/de"
    OUTPUT_FOLDER = "output"
    ocr_pdf_folder(INPUT_FOLDER, OUTPUT_FOLDER)
