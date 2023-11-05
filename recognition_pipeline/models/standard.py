import easyocr
from pytesseract import image_to_string


class BaseModel():
    def __init__(self, languages):
        self.languages = languages

    def predict(self, image):
        raise NotImplementedError
class EasyOcrModel(BaseModel):
    def __init__(self, languages):
        super().__init__(languages)
        self.model = easyocr.Reader(languages, gpu=False)

    def predict(self, image):
        return self.model.readtext(image)


class TesseractModel(BaseModel):
    def __init__(self, languages):
        super().__init__(languages)
        self.model = None
        lang_dict = {"de": "deu", "fr": "fra", "it": "ita", "en": "eng"}
        self.languages = "+".join([lang_dict[lang] for lang in languages])

    def predict(self, image):
        return image_to_string(image, lang=self.languages)