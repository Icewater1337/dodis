import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
class BertSummarizer():
    def __init__(self, model_name = 'mrm8488/bert2bert_shared-german-finetuned-summarization'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.model = EncoderDecoderModel.from_pretrained(model_name).to(self.device)

    def summarize(self, text):
        inputs = self.tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        output = self.model.generate(input_ids, attention_mask=attention_mask)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)