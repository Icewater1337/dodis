from pathlib import Path

from loguru import logger
from transformers import BartTokenizer, BartForConditionalGeneration
from utils.folder_util import load_file_contents_in_folder
import argparse, sys, os
from transformers import T5Tokenizer, T5ForConditionalGeneration,pipeline
from summarizer import Summarizer


# Summarize function
def summarize_bart(text, model, tokenizer):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=len(text)/10, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing the html files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where the text files should be saved")
    args = parser.parse_args()
    return args


def summarize_pipeline(file_content, summarizer):

    return summarizer(file_content, max_length=len(file_content)/10, min_length=100, do_sample=False, truncation=True)[0]['summary_text']

def summarize_t5(text, model, tokenizer):
    # Add the summarization prefix
    input_text = f"summarize: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
def summarize_distilbert(text, model):
    return model(text, num_sentences=5)


if __name__ == "__main__":
    parsed_args = parse_args()
    input_folder = parsed_args.input_folder
    output_folder = parsed_args.output_folder

    log_level = "TRACE"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    log_file = os.path.join(output_folder, "log_abstractive_summarizer.txt")
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add(log_file, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)

    logger.trace(f"Start loading files from {input_folder}")
    files_name_dict = load_file_contents_in_folder(input_folder, file_type="txt", return_dict=True)
    logger.trace(f"Finished loading files from {input_folder}")

    model_names = ["facebook/bart-large-cnn", "t5-small"]

    models = {"bart":{"model": BartTokenizer.from_pretrained("facebook/bart-large-cnn"), "tokenizer": BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")},
              "t5":{"model": T5Tokenizer.from_pretrained("t5-small"), "tokenizer": T5ForConditionalGeneration.from_pretrained("t5-small")}}



    #pipelines = {"bart": pipeline("summarization", model="facebook/bart-large-cnn"), "t5": pipeline("summarization", model="t5-small")}
    pipelines = {"t5": pipeline("summarization", model="t5-small")}

    for model_name, pipeline in pipelines.items():
        logger.trace(f"Start summarizing files using abstractive summarization model {model_name}")
        for name, file_content in files_name_dict.items():
            logger.trace(f"Summarizing file {name}")
            #summary = summarize_bart(file_content)
            summary = summarize_pipeline(file_content, pipeline)
            logger.success(f"Finished summarizing file {name}")
            output_folder_model = os.path.join(output_folder,"summaries", model_name)
            Path(output_folder_model).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_folder_model, name), "w", encoding="utf-8") as file:
                file.write(summary)


    logger.trace("start with extractive summarization using distilbert")

    # Load the model
    model = Summarizer('distilbert-base-uncased')
    for name, file_content in files_name_dict.items():
        logger.trace(f"Summarizing file {name}")
        summary = summarize_distilbert(file_content, model)
        logger.success(f"Finished summarizing file {name}")
        output_folder_model = os.path.join(output_folder,"summaries", "distilbert")
        Path(output_folder_model).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(output_folder_model, name), "w", encoding="utf-8") as file:
            file.write(summary)





