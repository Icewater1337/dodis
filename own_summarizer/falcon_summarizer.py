from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

class FalconSummarizer():
    def __init__(self, model_name="tiiuae/falcon-40b-instruct", temperature=0.1):
        model = model_name  # tiiuae/falcon-40b-instruct
        offload_folder = "/media/fuchs/d/huggingface_cache/models--tiiuae--falcon-40b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForCausalLM.from_pretrained(model,
                                                     torch_dtype=torch.bfloat16,
                                                     device_map="auto",
                                                     offload_folder=offload_folder
                                                     )

        pipe = transformers.pipeline(
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
        # llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
        llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temperature})
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Please summarize the text (in german) that follows and pay attention that you do not fantasize facts. \n{text}"
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def summarize(self, text_to_summarize):
        res = self.chain.run(text=text_to_summarize)
        return res
