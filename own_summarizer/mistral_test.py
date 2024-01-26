from llama_cpp import Llama

llm = Llama(model_path = "/media/fuchs/d/models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf")

prompt =  "Q: What are the names of the days of the week? A:"

output = llm(prompt)

print(output)