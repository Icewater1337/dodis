import easyocr
import numpy as np
from PIL import Image
from mmocr.apis import MMOCRInferencer, TextDetInferencer
from mmocr.utils import poly2bbox
from pytesseract import image_to_string, pytesseract, image_to_data, TesseractError
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
        try:
            return image_to_string(image, lang=self.languages)
        except TesseractError:
            return image_to_string(image)
        # return image_to_string(image)

class MMOCR(BaseModel):
    def __init__(self, det, rec, languages):
        super().__init__(languages)
        self.det = det
        self.rec = rec
        lang_dict = {"de": "deu", "fr": "fra", "it": "ita", "en": "eng"}
        self.languages = "+".join([lang_dict[lang] for lang in languages])
        self.model = MMOCRInferencer(det=self.det, rec=self.rec)

    def predict(self, image):
        output = self.model(str(image), show=False, print_result=True)

        return " ".join(output["predictions"][0]["rec_texts"])



class TrOCR(BaseModel):
    def __init__(self, languages):
        super().__init__(languages)
        self.model = None
        lang_dict = {"de": "deu", "fr": "fra", "it": "ita", "en": "eng"}
        self.languages = "+".join([lang_dict[lang] for lang in languages])
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        self.mmocr_detector =  TextDetInferencer(weights = "/home/fuchs/PycharmProjects/dodis/recognition_pipeline/dbnet_resnet50-oclip_1200e_icdar2015_20221102_115917-bde8c87a.pth")


    def extract_text_with_trocr(self,image):
        # Convert image to PIL format
        pil_img = Image.fromarray(np.uint8(image)).convert('RGB')
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text
    def predict(self, image):
        detection_results = self.mmocr_detector(np.array(image))
        word_boxes = [poly2bbox(polygon) for polygon in detection_results["predictions"][0]["polygons"]]
        texts = []
        img_numpy = np.array(image)
        for box in word_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            word_image = img_numpy[y_min:y_max, x_min:x_max]
            # Use TrOCR to OCR the cropped image
            text = self.extract_text_with_trocr(word_image)
            texts.append(text.lower())

        return " ".join(texts)
        # pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # generated_ids = self.model.generate(pixel_values)
        # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # return generated_text