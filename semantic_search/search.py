import argparse as argparse
import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer
import argparse
from utils.folder_util import load_file_contents_in_folder
import os

def search(query, embeddings, texts, top_k=5):
    inputs = tokenizer(query, return_tensors='pt', max_length=512, truncation=True).to(device)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    # Compute similarities
    similarities = [1 - cosine(query_embedding, emb) for emb in embeddings]

    # Sort by similarity
    sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)

    # Return top_k results
    return [(texts[i], similarities[i]) for i in sorted_indices[:top_k]]

def encode_texts(texts, model, tokenizer,device):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy())
    return embeddings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing the html files")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the folder where the text files should be saved")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    parsed_args = parse_args()
    input_folder = parsed_args.input_folder
    output_folder = parsed_args.output_folder

    text_dict = load_file_contents_in_folder(input_folder, "txt", return_dict=True)

    model_name = 'bert-base-german-cased'
    #model_name = 'tiiuae/falcon-180b'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model = model.to(device)

    embeddings_path = os.path.join(input_folder, "embeddings.npy")
    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path, allow_pickle=True)
    else:
        embeddings = encode_texts(list(text_dict.values()), model, tokenizer, device)
        np.save(embeddings_path, np.array(embeddings))



    while True:
        query = input("What are you searching for?")
        if query == "quit":
            break
    # Example search
        results = search(query, embeddings, list(text_dict.values()))
        for text, similarity in results:
            print(f"Similarity: {similarity}\nText: {text}\n")


#https://www.youtube.com/watch?v=qa0237G5ACY
# Dokument hinterlegen bei GPT und mit gpt-4 suchen.
# 1. Semantic search
# 2. conversational AI for questions. Konversation führen über dokument und fragen stellen. und nicht nur suche.
# 3. Landsacpe skizzieren von beiden dingen.
# Task oriented CAI (semantic serarch). Open CAI (open ended conversation)
# LLAMA falcon.
