from pathlib import Path
import openai
import torch
from loguru import logger
from transformers import BartTokenizer, BartForConditionalGeneration, AutoTokenizer

from own_summarizer.falcon_summarizer import FalconSummarizer
from own_summarizer.mistral_summarizer import MistralSummarizer
from own_summarizer.summary_quality_eval import check_grammar_and_spelling, evaluate_with_bertscore
from utils.folder_util import load_file_contents_in_folder
import argparse, sys, os
from transformers import T5Tokenizer, T5ForConditionalGeneration,pipeline
from summarizer import Summarizer
from gpt_summarizer import GptSummarizer as GPTSummarizer


# Summarize function
def summarize_bart(text, model, tokenizer):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=1024, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing the html files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where the text files should be saved")
    args = parser.parse_args()
    return args


def summarize_pipeline(file_content, summarizer):
    if torch.cuda.is_available():
        logger.trace("cuda is available")
    return summarizer(file_content, min_length=100, do_sample=False)[0]['summary_text']

def summarize_t5(text, model, tokenizer):
    # Add the summarization prefix
    input_text = f"summarize: {text}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
def summarize_distilbert(text, model):
    return model(text, num_sentences=5)


def evaluate_summary(summary, file_content):
    logger.trace("Evaluating quality of summary, using grammar check first")
    number_of_errors, errors = check_grammar_and_spelling(summary)
    logger.trace(f"Number of grammatical errors: {number_of_errors}")
    for error in errors:
        logger.trace(error)
    logger.trace("Evaluating quality of summary, using bertscore")
    precision, recall, f1 = evaluate_with_bertscore([summary], [file_content])
    logger.trace(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")


if __name__ == "__main__":
    openai.api_key = os.environ["OPENAI_API_KEY"]
    parsed_args = parse_args()
    input_folder = parsed_args.input_folder
    output_folder = parsed_args.output_folder
    access_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-40B", access_token=access_token)
    log_level = "TRACE"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"

    log_file = os.path.join(output_folder, "log_abstractive_summarizer.txt")
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add(log_file, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)

    logger.trace(f"Start loading files from {input_folder}")
    files_name_dict = load_file_contents_in_folder(input_folder, file_type="txt", return_dict=True)
    logger.trace(f"Finished loading files from {input_folder}")

    # model_names = ["facebook/bart-large-cnn", "t5-small", "tiiuae/falcon-40B"]
    # model_names = ["tiiuae/falcon-40B"]

    # models = {"bart":{"model": BartTokenizer.from_pretrained("facebook/bart-large-cnn"), "tokenizer": BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")},
    #           "t5":{"model": T5Tokenizer.from_pretrained("t5-small"), "tokenizer": T5ForConditionalGeneration.from_pretrained("t5-small")},
    #           "falcon":{"model": T5Tokenizer.from_pretrained("tiiuae/falcon-40B"), "tokenizer":  AutoTokenizer.from_pretrained("tiiuae/falcon-40B", access_token=access_token)}}


    device = 0 if torch.cuda.is_available() else -1

    # pipelines = {"bart": pipeline("summarization", model="facebook/bart-large-cnn",  device=device),
    #              "t5": pipeline("summarization", model="t5-small",  device=device)
    #              }

    # pipelines = {}
    # for model_name, pipeline in pipelines.items():
    #     logger.trace(f"Start summarizing files using abstractive summarization model {model_name}")
    #     for name, file_content in files_name_dict.items():
    #         logger.trace(f"Summarizing file {name}")
    #         #summary = summarize_bart(file_content)
    #         summary = summarize_pipeline(file_content, pipeline)
    #         logger.success(f"Finished summarizing file {name}")
    #
    #         evaluate_summary(summary, file_content)
    #
    #         output_folder_model = os.path.join(output_folder,"summaries", model_name)
    #         Path(output_folder_model).mkdir(parents=True, exist_ok=True)
    #         with open(os.path.join(output_folder_model, name), "w", encoding="utf-8") as file:
    #             file.write(summary)

    # logger.trace("start with extractive summarization using distilbert")
    #
    # # Load the model
    # model = Summarizer('distilbert-base-uncased')
    #
    # for name, file_content in files_name_dict.items():
    #     logger.trace(f"Summarizing file {name}")
    #     summary = summarize_distilbert(file_content, model)
    #     logger.success(f"Finished summarizing file {name}")
    #     evaluate_summary(summary, file_content)
    #     output_folder_model = os.path.join(output_folder,"summaries", "distilbert")
    #     Path(output_folder_model).mkdir(parents=True, exist_ok=True)
    #     with open(os.path.join(output_folder_model, name), "w", encoding="utf-8") as file:
    #         file.write(summary)

    # llm_summarizers = [MistralSummarizer(), GPTSummarizer()]
    llm_summarizers = [MistralSummarizer()]
    # llm_summarizers = [GPTSummarizer()]
    # gpt_summarizer = GPTSummarizer()
    for summi in llm_summarizers:
        logger.trace(f"start with abstractrive summarization using {summi.__class__.__name__}")

        for name, file_content in files_name_dict.items():
            logger.trace(f"Summarizing file {name}")
            # summary = summarize_bart(file_content)
            summary = summi.summarize(file_content)
            logger.success(f"Finished summarizing file {name}")

            evaluate_summary(summary, file_content)

            output_folder_model = os.path.join(output_folder, "summaries", summi.__class__.__name__)
            Path(output_folder_model).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(output_folder_model, name), "w", encoding="utf-8") as file:
                file.write(summary)





# ToDO:
# 3 Dokumente, kurze, mittlere, lange, Für jedes ein eigenes summary erstellen.
# Immer gleiche anzahl tokens für alle modelle
# 1. Tabelle mit allen toools und welche issues dass es gibt (vor/ nachteile). z.B. beliebig lange summaries machbar (checkbox).
# 2. Tabelle mit fairen comparison scores. 1. Grammatikalisch, 2. Fakten, 3. VOllständigkeit
# Get 3 Abstractive, get 3 generative summaries.

