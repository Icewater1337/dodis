import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from recognition_pipeline.models.standard import TesseractModel, EasyOcrModel, TrOCR
from loguru import logger
from torchmetrics.text import CharErrorRate, WordErrorRate


def calculate_accuracy(transcript_txt, text):
    cer = CharErrorRate()
    cer_res = cer(text, transcript_txt)

    wer = WordErrorRate()
    wer_res = wer(text, transcript_txt)

    return cer_res.item(), wer_res.item()


def process_file(model, file, txt_folder):
    logger.trace(f"Start with file: {file}")
    image = Image.open(file)
    if isinstance(model, TrOCR):
        image = image.convert("RGB")
    else:
        image = preprocess_image(image)
    transcript_path = Path(txt_folder) / file.with_suffix(".txt").name
    transcript_txt = transcript_path.read_text(encoding="utf-8")

    predicted_text = model.predict(image)
    logger.trace(f"Text: {predicted_text} for file: {file}")
    cer, wer = calculate_accuracy(transcript_txt, predicted_text)
    logger.success(f"File: {file} | CER: {cer} | WER: {wer}")

    return {
        "filename": file.name,
        "CER": cer,
        "WER": wer,
        "predicted": predicted_text,
        "actual": transcript_txt
    }

def evaluate_model(model_name, languages, output_folder, img_folder, txt_folder):
    model_dict = {"tesseract": TesseractModel, "easyocr": EasyOcrModel, "trocr": TrOCR}
    model = model_dict[model_name](languages)
    output_file_path = Path(output_folder) / f"results_{model_name}.csv"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    # Prepare arguments for starmap
    file_paths = [(model, file, txt_folder) for file in Path(img_folder).iterdir()]
    # file_paths = file_paths[:100]
    results = []
    # Using multiprocessing Pool to process files in parallel
    # with Pool() as pool:
    #     results = pool.starmap(process_file, file_paths)
    for file_path in file_paths:
        results.append(process_file(*file_path))
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_file_path, index=False)
    calculate_dataset_wer_cer(results_df)


    return results_df
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

'''
Take a given dataframe with individual CER and WER for each picture.
Calculate the average CER and WER for the whole dataset.
'''
def calculate_dataset_wer_cer(dataframe):
    dataframe = dataframe[dataframe["CER"] != np.inf]
    dataframe = dataframe[dataframe["WER"] != np.inf]
    filtered_cer = dataframe[dataframe["CER"] < 0.5]
    filtered_wer = dataframe[dataframe["WER"] < 0.5]

    outliers = dataframe[dataframe["CER"] > 0.8]

    # filter out infinity
    avg_cer = filtered_cer["CER"].mean()
    avg_wer = filtered_wer["WER"].mean()

    logger.success(f"Filtered Average CER: {avg_cer} | Average WER: {avg_wer}")
    avg_cer_unfiltered = dataframe["CER"].mean()
    avg_wer_unfiltered = dataframe["WER"].mean()
    logger.success(f"Unfiltered Average CER: {avg_cer_unfiltered} | Average WER: {avg_wer_unfiltered}")

if __name__ == "__main__":
    # models = ["tesseract", "easyocr", "trocr"]
    # models = ["easyocr"]
    models = ["trocr"]
    languages = ["de", "fr", "it", "en"]
    output_folder = "/media/fuchs/d/dataset_try_4/final_dataset/output/"
    img_folder = "/media/fuchs/d/dataset_try_4/final_dataset/png/"
    txt_folder = "/media/fuchs/d/dataset_try_4/final_dataset/txt/"
    log_file = "/media/fuchs/d/dataset_try_4/final_dataset/output/log.txt"

    log_level = "TRACE"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add(log_file, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)


    for model in models:
        logger.trace(f"Start with Model: {model} and languages: {languages}")
        evaluate_model(model, languages, output_folder, img_folder, txt_folder)
        logger.success(f"Finished with Model: {model} and languages: {languages}")
