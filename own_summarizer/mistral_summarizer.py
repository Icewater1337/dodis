from langchain import HuggingFacePipeline, PromptTemplate, LLMChain

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

class MistralSummarizer():
    def __init__(self, model_name="mistralai/Mixtral-8x7B-v0.1", temperature=0.1):
        model = model_name  # tiiuae/falcon-40b-instruct
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model)

        self.text = "Bitte fasse mir das nachfolgende Dokument Zusammen. Bitte achte auch darauf, dass es nicht länger ist als {zeichen}\n {text}"



        # offload_folder = "/media/fuchs/d/huggingface_cache/models--tiiuae--falcon-40b-instruct"
        # tokenizer = AutoTokenizer.from_pretrained(model)
        # model = AutoModelForCausalLM.from_pretrained(model,
        #                                              torch_dtype=torch.bfloat16,
        #                                              device_map="auto",
        #                                              offload_folder=offload_folder
        #                                              )
        #
        self.pipe = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16        )


        # # llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
        # llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':temperature})
        # prompt = PromptTemplate(
        #     input_variables=["text"],
        #     template="Please summarize the text (in german) that follows and pay attention that you do not fantasize facts. \n{text}"
        # )
        # self.chain = LLMChain(llm=llm, prompt=prompt)

    def summarize(self, text_to_summarize):
        text = self.text.format(zeichen=len(text_to_summarize), text=text_to_summarize)
        sequences = self.pipe(
            text,
            do_sample=True,
            max_new_tokens=100,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
        )
        return sequences[0]["generated_text"]
        # inputs = self.tokenizer(self.text + text_to_summarize, return_tensors="pt")
        #
        # outputs = self.model.generate(**inputs, max_new_tokens=20)
        # res = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # print(res)
        # return res
