import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from recognition_pipeline.models.standard import TesseractModel, EasyOcrModel
from loguru import logger
from torchmetrics.text import CharErrorRate, WordErrorRate


def calculate_accuracy(transcript_txt, text):
    cer = CharErrorRate()
    cer_res = cer(text, transcript_txt)

    wer = WordErrorRate()
    wer_res = wer(text, transcript_txt)

    return cer_res.item(), wer_res.item()


def evaluate_model(model_name, languages, output_folder, img_folder, txt_folder):
    model_dict = {"tesseract": TesseractModel, "easyocr": EasyOcrModel}
    model = model_dict[model_name](languages)
    results = []  # List to hold dictionaries with the results
    output_file_path = Path(output_folder) / f"results_{model_name}.csv"


    for file in Path(img_folder).iterdir():
        logger.trace(f"Start with file: {file}")
        image = Image.open(file)
        image = preprocess_image(image)
        transcript_path = Path(txt_folder) / file.with_suffix(".txt").name
        transcript_txt = transcript_path.read_text(encoding="utf-8")

        predicted_text = model.predict(image)
        logger.trace(f"Text: {predicted_text} for file: {file}")
        cer, wer = calculate_accuracy(transcript_txt, predicted_text)
        logger.success(f"File: {file} | CER: {cer} | WER: {wer}")

        # Append a dictionary with results for the current file to the results list
        results.append({
            "filename": file.name,
            "CER": cer,
            "WER": wer,
            "predicted": predicted_text,
            "actual": transcript_txt
        })

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)


    # Save the DataFrame to a CSV file
    results_df.to_csv(output_file_path, index=False)

    return results_df  # Optionally return the DataFrame

def preprocess_image(image):
    logger.trace(f"Preprocessing image {image}")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if 45 < angle <= 90:
        angle = angle - 90
    if not (-45 <= angle <= 45):  # The text should not be skewed more than 45 degrees.
        angle = 0

    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return image

if __name__ == "__main__":
    # models = ["tesseract", "easyocr"]
    models = ["easyocr"]
    languages = ["de", "fr", "it", "en"]
    output_folder = "/media/fuchs/d/dataset_try_2/final_dataset/output/"
    img_folder = "/media/fuchs/d/dataset_try_2/final_dataset/png/"
    txt_folder = "/media/fuchs/d/dataset_try_2/final_dataset/txt/"
    log_file = "/media/fuchs/d/dataset_try_2/final_dataset/output/log.txt"

    log_level = "TRACE"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add(log_file, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)


    for model in models:
        logger.trace(f"Start with Model: {model} and languages: {languages}")
        evaluate_model(model, languages, output_folder, img_folder, txt_folder)
