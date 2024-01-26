from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class GptSummarizer():
    def __init__(self, model_name="gpt-4-1106-preview", temperature=0.1):
        # llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
        llm = ChatOpenAI(temperature=temperature, model_name=model_name)
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Please summarize the text (in german) that follows and pay attention that you do not fantasize facts. \n{text}"
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def summarize(self, text_to_summarize):
        res = self.chain.run(text=text_to_summarize)
        return res
