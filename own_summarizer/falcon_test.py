from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-40b-instruct"
offload_folder = "/media/fuchs/d/huggingface_cache/models--tiiuae--falcon-40b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto",
                                             offload_folder = offload_folder
                                             )

pipeline = transformers.pipeline(
            "text-generation",  # task
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )