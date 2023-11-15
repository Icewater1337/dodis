import easyocr
import numpy as np
from PIL import Image
from pytesseract import image_to_string, pytesseract, image_to_data
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


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
        result =  self.model.readtext(image)
        extracted_text = ''

        # Iterate over the result
        for detection in result:
            text = detection[1]
            extracted_text += text + ' '

        # Remove the trailing space
        extracted_text = extracted_text.strip()
        return extracted_text


class TesseractModel(BaseModel):
    def __init__(self, languages):
        super().__init__(languages)
        self.model = None
        lang_dict = {"de": "deu", "fr": "fra", "it": "ita", "en": "eng"}
        self.languages = "+".join([lang_dict[lang] for lang in languages])

    def predict(self, image):
        return image_to_string(image, lang=self.languages)


class TrOCR(BaseModel):
    def __init__(self, languages):
        super().__init__(languages)
        self.model = None
        lang_dict = {"de": "deu", "fr": "fra", "it": "ita", "en": "eng"}
        self.languages = "+".join([lang_dict[lang] for lang in languages])
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    def extract_text_with_trocr(self,image):
        # Convert image to PIL format
        pil_img = Image.fromarray(np.uint8(image)).convert('RGB')
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    def predict(self, image):
        tessdata = image_to_data(image,output_type=pytesseract.Output.DICT)
        texts = []
        img_numpy = np.array(image)
        n_boxes = len(tessdata['level'])
        for i in range(n_boxes):
            if int(tessdata['conf'][i]) > 0:  # Confidence level > 0 to ensure it's valid text
                (x, y, w, h) = (tessdata['left'][i], tessdata['top'][i], tessdata['width'][i], tessdata['height'][i])
                cropped_image = img_numpy[y:y + h, x:x + w]

                # Use TrOCR to OCR the cropped image
                text = self.extract_text_with_trocr(cropped_image)
                texts.append(text.lower())

        return " ".join(texts)
        # pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # generated_ids = self.model.generate(pixel_values)
        # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return generated_text