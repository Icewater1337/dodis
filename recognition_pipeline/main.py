import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt

from recognition_pipeline.models.standard import TesseractModel, EasyOcrModel, TrOCR, MMOCR
from loguru import logger
from torchmetrics.text import CharErrorRate, WordErrorRate


def calculate_accuracy(transcript_txt, text):
    cer = CharErrorRate()
    cer_res = cer(text, transcript_txt)

    wer = WordErrorRate()
    wer_res = wer(text, transcript_txt)

    return cer_res.item(), wer_res.item()

def deskew(image):
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

def binarize(image):
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    return thresh

def denoise_dilate(image):
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    return image

def denoise_gaussian(image):
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def thinning(image):
    # thinned = cv2.ximgproc.thinning(image, image, cv2.ximgproc.THINNING_ZHANGSUEN)
    # thinned = cv2.ximgproc.thinning(image)
    inverted_image = cv2.bitwise_not(image)

    thinned_image = cv2.ximgproc.thinning(inverted_image, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    thinned_image = cv2.bitwise_not(thinned_image)

    # kernel = np.ones((5, 5), np.uint8)
    # erosion = cv2.erode(image, kernel, iterations=1)
    return thinned_image

def process_file(model, file, txt_folder, preprocess_steps):
    logger.trace(f"Start with file: {file}")
    image = Image.open(file)
    # if isinstance(model, TrOCR):
    # image = image.convert("RGB")
    # else:
    image = preprocess_image_pipeline(image, preprocess_steps)
    # image = preprocess_image(image)
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

def evaluate_model(model, model_name, languages, output_folder, img_folder, txt_folder, preprocess_steps = None, preprocess_name = ""):

    output_file_path = Path(output_folder) / f"results_{model_name}_{preprocess_name}.csv"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    # Prepare arguments for starmap
    file_paths = [(model, file, txt_folder, preprocess_steps) for file in Path(img_folder).iterdir()]
    # file_paths = [(model, Path("/media/fuchs/d/dataset_try_5/final_dataset/png/42887_007_de.png"), txt_folder, preprocess_steps)]
    # file_paths = file_paths[:100]
    results = []
    # Using multiprocessing Pool to process files in parallel
    # with Pool() as pool:
    #     results = pool.starmap(process_file, file_paths)
    for file_path in file_paths:
        # if not file_path[1].name == "46321_003_de.png":
        #     continue
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

def preprocess_image_pipeline(image, pipeline):
    logger.trace(f"Preprocessing image {image}")
    # opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    np_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    for step in pipeline:
        np_image = step(np_image)
        # plt.imshow(np_image, cmap='gray')
        # plt.axis('off')
        # plt.show()
        # plt.clf()
    return np_image
'''
Take a given dataframe with individual CER and WER for each picture.
Calculate the average CER and WER for the whole dataset.
'''
def calculate_dataset_wer_cer(dataframe):
    dataframe = dataframe[dataframe["CER"] != np.inf]
    dataframe = dataframe[dataframe["WER"] != np.inf]
    # dataframe = dataframe[dataframe["CER"] < 0.9]
    # dataframe = dataframe[dataframe["WER"] < 0.9]
    dataframe = dataframe[(dataframe['CER'] <= 0.5) & (dataframe['WER'] <= 0.9)]

    outliers = dataframe[dataframe["CER"] > 0.8]

    # filter out infinity
    avg_cer = dataframe["CER"].mean()
    avg_wer = dataframe["WER"].mean()

    logger.success(f"Filtered Average CER: {avg_cer} | Average WER: {avg_wer}")
    avg_cer_unfiltered = dataframe["CER"].mean()
    avg_wer_unfiltered = dataframe["WER"].mean()
    logger.success(f"Unfiltered Average CER: {avg_cer_unfiltered} | Average WER: {avg_wer_unfiltered}")

if __name__ == "__main__":
    # models = ["tesseract", "easyocr", "trocr", "mmocr"]
    # model_names = ["tesseract", "easyocr"]
    models = []
    # preprocess_steps = [binarize, deskew, denoise_dilate, denoise_gaussian, thinning]
    preprocess_steps_full = {"nothing": [],
                             "bin+ deskew": [binarize, deskew],"bin+thin": [binarize, thinning],
                             "bin+denoiose": [binarize, denoise_gaussian],
                             "bin+deskew+thin+denoise": [binarize, deskew, thinning, denoise_gaussian],
                             }

    # preprocess_steps = [binarize, deskew, thinning, denoise_gaussian]
    model_names = ["tesseract"]
    languages = ["de", "fr", "it", "en"]
    # languages = ["fr"]
    full_model_names = []
    output_folder = "/media/fuchs/d/dataset_try_5/final_dataset/output/"
    img_folder = "/media/fuchs/d/dataset_try_5/final_dataset/png/"
    txt_folder = "/media/fuchs/d/dataset_try_5/final_dataset/txt/"

    # mmocr_det = ["DBNet", "DBNetpp", "PANet", "TextSnake"]
    # mmocr_det = ["DBNet",  "DBNetpp"]
    mmocr_det = ["DBNet"]
    # mmocr_rec = ["ABINet", "CRNN", "NRTR", "SAR", "SegOCR", "STARNet", "RobustScanner"]
    mmocr_rec = ["CRNN"]
    model_dict = {"tesseract": TesseractModel, "easyocr": EasyOcrModel, "trocr": TrOCR}
    log_level = "TRACE"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    for model_name in model_names:
        if model_name == "mmocr":
            for det in mmocr_det:
                for rec in mmocr_rec:
                    full_model_names.append(f"mmocr_{det}_{rec}")
                    models.append(MMOCR(det, rec, languages))
        else:
            model = model_dict[model_name]
            models.append(model(languages))
            full_model_names.append(model_name)


    for model, model_name in zip(models, full_model_names):
        for preprocess_name, preprocess_steps in preprocess_steps_full.items():
            log_file = f"/media/fuchs/d/dataset_try_5/final_dataset/output/log_{model_name}_{preprocess_name}.txt"
            logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
            logger.add(log_file, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
            logger.trace(f"Start with Model: {model} and languages: {languages}")
            logger.trace(f"Start with preprocessing: {preprocess_name}")
            # result_df = pd.read_csv(Path(output_folder) / f"results_{model_name}.csv")
            # calculate_dataset_wer_cer(result_df)
            evaluate_model(model, model_name, languages, output_folder, img_folder, txt_folder, preprocess_steps = preprocess_steps, preprocess_name = preprocess_name)
            logger.success(f"Finished with Model: {model} and languages: {languages}")
