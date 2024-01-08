from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class Summarizer():
    def __init__(self):
        # llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
        llm = ChatOpenAI(temperature=0.1, model_name="gpt-4-1106-preview")
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Nachfolgend findest du ein historisches Dokument. Bitte fasse das Dokument zusammen und achte dich besonders darauf, dass du beim zusammenfassen"
                     "die Fakten nicht veränderst. Der Inhalt ist sehr wichtig: {text}. \n\n"
        )
        self.chain = LLMChain(llm=llm, prompt=prompt)

    def summarize(self, text_to_summarize):
        res = self.chain.run(text=text_to_summarize)
        return res
