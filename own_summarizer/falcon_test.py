import os
import subprocess
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

from utils.folder_util import load_file_contents_in_folder

#from huggingface_hub import login
#login()


model = "tiiuae/falcon-180B"


access_token = os.environ["HUGGINGFACE_ACCESS_TOKEN"]
files_name_dict = load_file_contents_in_folder("../inputs_short/", file_type="txt", return_dict=True)
text = list(files_name_dict.values())[0]
tokenizer = AutoTokenizer.from_pretrained(model, access_token=access_token)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
   f"Nachfolgend findest du ein historisches Dokument. Bitte fasse das Dokument zusammen und achte dich besonders darauf, dass du beim zusammenfassen die Fakten nicht veränderst. Der Inhalt ist sehr wichtig: {text}.",
    max_length=200,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
