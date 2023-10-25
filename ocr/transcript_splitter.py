import os
from pathlib import Path

import cv2
import easyocr
import layoutparser as lp
import numpy as np
import pytesseract
from PIL import ImageOps
from fuzzywuzzy import fuzz
from matplotlib import pyplot as plt, patches
from numpy import argmin, argmax
from pdf2image import convert_from_path
from pytesseract import image_to_string, image_to_data


def layout_parser_test(img):
    model = lp.Detectron2LayoutModel(
        config_path='/home/fuchs/.torch/iopath_cache/s/dgy9c10wykk4lq4/config.yaml',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
    )
    model.cfg.MODEL.WEIGHTS = '/home/fuchs/.torch/iopath_cache/s/dgy9c10wykk4lq4/model_final.pth'
    # Load the image
    img = np.array(img)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect layout
    layout = model.detect(image)

    # Find the block with the largest area
    largest_block = max(layout, key=lambda x: x.width * x.height)
    # Crop the image to the largest block
    x1, y1, x2, y2 = largest_block.coordinates
    cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

    # Plot the cropped image
    # plt.imshow(cropped_image)
    # plt.axis('off')  # to hide the axis values
    # plt.show()

    # Extract text content using OCR

    return cropped_image

def find_matching_substring(ocrd, transcript):
    ocrd_words = ocrd.split()
    transcript_words = transcript.split()

    for i in range(len(ocrd_words)):
        for j in range(len(transcript_words)):
            if ocrd_words[i] == transcript_words[j]:
                k = 1
                while (i + k < len(ocrd_words) and j + k < len(transcript_words)) and ocrd_words[i + k] == \
                        transcript_words[j + k]:
                    k += 1
                if k > 5:  # let's assume we need at least 6 words in sequence to consider it a match
                    return ' '.join(ocrd_words[i:i + k])
    return None


def find_matching_index_front(ocrd_words, transcript):
    # ocrd_words = ocrd.split()
    transcript_words = transcript.split()
    inexact = False
    # Look for exact match
    for j in range(len(ocrd_words) - 2):
        for i in range(len(transcript_words) - 2):
            if transcript_words[i:i + 3] == ocrd_words[j:j + 3]:
                return j, i, inexact

    print("No exact match found. Attempting fuzzy match...")
    inexact = True
    max_ratio = 0
    index = -1
    for j in range(len(ocrd_words) - 2):
        for i in range(len(transcript_words) - 2):
            ratio = fuzz.ratio(" ".join(transcript_words[i:i + 3]), " ".join(ocrd_words[j:j + 3]))
            if ratio > max_ratio:
                max_ratio = ratio
                index = i
    if max_ratio > 80:  # Threshold for approximate match. Adjust as needed.
        return j, index, inexact

    return -1, -1,inexact  # Return -1 if no match found
def find_matching_index_back(ocrd, transcript):
    ocrd_words = ocrd[::-1]  # Reverse the list
    transcript_words = transcript[::-1]  # Reverse the list
    inexact = False
    # Look for exact match but only in the first 30 words
    for j in range(max(30,len(ocrd_words) - 2)):
        for i in range(len(transcript_words) - 2):
            if transcript_words[i:i + 3] == ocrd_words[j:j + 3]:
                return len(ocrd_words) - j - 1, len(transcript_words) - i - 1, inexact  # Convert index back to original order

    print("No exact match found. Attempting fuzzy match...")
    inexact = True
    max_ratio = 0
    index = -1
    for j in range(len(ocrd_words) - 2):
        for i in range(len(transcript_words) - 2):
            ratio = fuzz.ratio(" ".join(transcript_words[i:i + 3]), " ".join(ocrd_words[j:j + 3]))
            if ratio > max_ratio:
                max_ratio = ratio
                index = i
    if max_ratio > 80:  # Threshold for approximate match. Adjust as needed.
        return len(ocrd_words) - j - 1, len(transcript_words) - index - 1, inexact # Convert index back to original order

    return -1, -1, inexact
def convert_pdf_to_images(pdf_path):
    print(f"Trying to load {pdf_path}")
    try:
        images = convert_from_path(pdf_path)
    except:
        images = []
        print(f"Could not load {pdf_path}. Skipping file")
    return images

def auto_crop(image):
    # Convert the image to grayscale
    grayscale = image.convert('L')
    # Use numpy to find the bounding box of the content
    bbox = np.array(grayscale).nonzero()
    rows, cols = bbox
    rectangle = min(rows), min(cols), max(rows), max(cols)
    # Crop the image based on the bounding box
    cropped_image = image.crop(rectangle)
    return cropped_image


def merge_boxes(boxes, threshold=50):
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[1])  # Sort by top y-coordinate

    merged = []
    current_box = boxes[0]
    x_current, y_current, w_current, h_current = current_box

    for box in boxes[1:]:
        x_next, y_next, w_next, h_next = box

        if y_next - (y_current + h_current) < threshold:  # Check if boxes are close vertically
            x_combined = min(x_current, x_next)
            y_combined = y_current
            w_combined = max(x_current + w_current, x_next + w_next) - x_combined
            h_combined = y_next + h_next - y_current

            current_box = (x_combined, y_combined, w_combined, h_combined)
            x_current, y_current, w_current, h_current = current_box
        else:
            merged.append(current_box)
            current_box = box
            x_current, y_current, w_current, h_current = box

    merged.append(current_box)
    return merged

def choose_main_text_block(boxes, image_width, image_height):
    max_area = 0
    main_text_block = None

    for box in boxes:
        x, y, w, h = box
        area = w * h

        # You can add more conditions based on position, aspect ratio, etc. if needed
        # Here, we give a bias towards boxes located towards the center of the image
        distance_to_center = abs(image_width/2 - (x + w/2)) + abs(image_height/2 - (y + h/2))

        if area > max_area and distance_to_center < image_width/2:
            max_area = area
            main_text_block = box

    return main_text_block


def find_text_block_2(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    exclude_top_percentage = 0.1
    start_y = int(exclude_top_percentage * gray.shape[0])

    gray = gray[start_y:, :]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # kernel = np.ones((5, 5), np.uint8)
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1500
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    if not valid_contours:
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
        return

    boxes = [cv2.boundingRect(contour) for contour in valid_contours]
    merged_boxes_list = merge_boxes(boxes)

    main_text_block = choose_main_text_block(merged_boxes_list, img.shape[1], img.shape[0])
    x, y, w, h = main_text_block
    padding = 170
    x_min = max(0,x-padding)
    y_min = max(0,y-padding)
    x_max = min(img.shape[1],x+w+padding)
    y_max = min(img.shape[0],y+h+padding)
    # cropped = img[y:y + h, x:x + w]
    cropped = img[y_min:y_max, x_min:x_max]
    # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    # plt.show()
    return cropped

def find_text_block(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Exclude the top 30% of the image
    exclude_top_percentage = 0.1
    start_y = int(exclude_top_percentage * gray.shape[0])



    # Remaining image after excluding top
    # gray = gray[start_y:, :]
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    # plt.title("Image after Excluding Top Part")
    # plt.show()

    gray_cropped = gray[start_y:, :]
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray_cropped, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((5, 5), np.uint8)
    # Dilate to connect broken parts
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1000
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    if not valid_contours:
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
        return

    boxes = [cv2.boundingRect(contour) for contour in valid_contours]
    merged_boxes_list = merge_boxes(boxes)  # Assuming you have this function

    main_text_block = choose_main_text_block(merged_boxes_list, img.shape[1],
                                             img.shape[0])  # Assuming you have this function
    x, y, w, h = main_text_block
    padding = 10
    x_min = max(0, x - padding)
    y_min = max(0, y - padding + start_y)  # Adjust the y-coordinate with the excluded portion
    x_max = min(img.shape[1], x + w + padding)
    y_max = min(img.shape[0], y + h + padding + start_y)  # Adjust the y-coordinate with the excluded portion
    cropped = img[y_min:y_max, x_min:x_max]

    # plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    # plt.title("Cropped Text Block")
    # plt.show()

    return cropped



def extract_text_from_image(image_path):
    return image_to_string(image_path)

def extract_data_from_image(image):
    return image_to_data(image,output_type=pytesseract.Output.DICT)
def extract_text_from_image_easyocr(image):
    reader = easyocr.Reader(['de'])
    results = reader.readtext(np.array(image))
    all_text = ""
    for (_, text, _) in results:
        all_text += " " + text
    return all_text.strip()

def find_closest_match(target, source):
    print("No match found, do fuzzy")
    highest_score = -1
    match_idx = -1
    for i in range(len(source) - len(target) + 1):
        current_score = fuzz.ratio(" ".join(target), " ".join(source[i:i+len(target)]))
        if current_score > highest_score:
            highest_score = current_score
            match_idx = i
    # Max match fuzzy is 100. We could se 50 as threshold. And see that
    # we return -1 if no match is found and else the index but of the letters, not array

    return match_idx


def remove_dodis(ocr_words):
    for i in range(len(ocr_words)):
        if ocr_words[i].lower() == "dodis":
            return ocr_words[:i]
    return ocr_words
    # idx_dodis = ocr_words.index("dodis")
    # return ocr_words[:idx_dodis]
#
def text_alignment_match(matched_transcript, image):
    pass


def extract_corresponding_transcript(ocr_words, full_transcript):
    # ocr_words = ocr_text.split()

    transcript_words = full_transcript.split()

    ocr_idx, transcript_idx, inexact_start = find_matching_index_front(ocr_words, full_transcript)

    if ocr_idx == -1 or transcript_idx == -1:
        return None, full_transcript, True, []

    ocr_words = ocr_words[ocr_idx:]
    transcript_words = transcript_words[transcript_idx:]
    ocr_words = remove_dodis(ocr_words)

    ocr_end_index, transcript_end_idx, inexact_end = find_matching_index_back(ocr_words, transcript_words)

    relevant_indices = list(range(ocr_idx, ocr_end_index+ocr_idx + 1))

    if ocr_end_index == -1 or transcript_end_idx == -1:
        return None, full_transcript, True, []

    # Extract the corresponding text from the transcript
    matched_text = ' '.join(transcript_words[:transcript_end_idx+1])
    remaining_transcript = ' '.join(transcript_words[transcript_end_idx+1:])
    inexact = inexact_start or inexact_end

    return matched_text, remaining_transcript, inexact, relevant_indices


def remove_white_border(img):
    gray_img = img.convert('L')
    binary_img = gray_img.point(lambda x: 0 if x < 200 else 255, '1')
    inverted_img = ImageOps.invert(binary_img)
    bbox = inverted_img.getbbox()
    cropped_img = img.crop(bbox)
    return cropped_img


def remove_top_part(image):
    exclude_top_percentage = 0.2
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    start_y = int(exclude_top_percentage * gray.shape[0])

    # gray = gray[start_y:, :]

    img_cropped = np.array(image)[start_y:, :]
    # plt.imshow(cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))
    # plt.title("Image after Excluding Top Part")
    # plt.show()
    return img_cropped


def preprocess_image(image, is_first_page):
    image = remove_white_border(image)
    if is_first_page:
        image = remove_top_part(image)
    return np.array(image)


def backcrop_image(image, indices_to_keep, tessdata, take_out_parts = False):
    plots = True
    # Initial boundaries of the words to keep
    word_boxes = [(
        tessdata["left"][index],
        tessdata["top"][index],
        tessdata["left"][index] + tessdata["width"][index],
        tessdata["top"][index] + tessdata["height"][index]
    ) for index in indices_to_keep]

    heights = [(box[3] - box[1]) for box in word_boxes]
    average_height = sum(heights) / len(heights)

    word_boxes = [box for box in word_boxes if (box[3] - box[1]) <= 2 * average_height]

    crop_x1 = min([box[0] for box in word_boxes])
    idx_x1 = argmin([box[0] for box in word_boxes])
    crop_y1 = min([box[1] for box in word_boxes])
    idx_y1 = argmin([box[1] for box in word_boxes])
    crop_x2 = max([box[2] for box in word_boxes])
    idx_x2 = argmax([box[2] for box in word_boxes])
    crop_y2 = max([box[3] for box in word_boxes])
    idx_y2 = argmax([box[3] for box in word_boxes])

    # Copy the original image for manipulation
    masked_image = image.copy()

    if take_out_parts:
    # Black out the interfering boxes
        for index in set(range(len(tessdata["left"]))) - set(indices_to_keep):
            x1, y1, x2, y2 = (
                tessdata["left"][index],
                tessdata["top"][index],
                tessdata["left"][index] + tessdata["width"][index],
                tessdata["top"][index] + tessdata["height"][index]
            )
            if (crop_x1 < x1 < crop_x2 and crop_x1 < x2 < crop_x2) and (crop_y1 < y1 < crop_y2 and crop_y1 < y2 < crop_y2):
                masked_image[y1:y2, x1:x2] = 255  # assuming white background, adjust accordingly

    # Add padding
    padding = 3
    crop_x1 = 0
    # crop_x1 = max(0, crop_x1 - padding)
    crop_y1 = max(0, crop_y1 - padding)
    crop_x2 = image.shape[1]
    # crop_x2 = min(image.shape[1], crop_x2 + padding)
    crop_y2 = min(image.shape[0], crop_y2 + padding)

    # Crop the image
    cropped_masked_image = masked_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if plots:
        # Plot the original image with bounding boxes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        for box in word_boxes[-10:]:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        box_max_x = word_boxes[idx_x2]
        rect_max_x = patches.Rectangle((box_max_x[0], box_max_x[1]), box_max_x[2] - box_max_x[0], box_max_x[3] - box_max_x[1], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_max_x)

        box_max_y = word_boxes[idx_y2]
        rect_max_y = patches.Rectangle((box_max_y[0], box_max_y[1]), box_max_y[2] - box_max_y[0], box_max_y[3] - box_max_y[1], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_max_y)

        plt.title("Original Image with Bounding Boxes")
        plt.axis('off')
        plt.show()
        plt.imshow(cropped_masked_image)
        plt.axis('off')
        plt.show()

    return cropped_masked_image


def main(pdf_path,text_path,output_folder):
    with open(text_path, 'r') as f:
        full_transcript = f.read()

    images = convert_pdf_to_images(pdf_path)
    # images = convert_from_path(pdf_path, dpi=300)
    # images = [auto_crop(img) for img in images]

    txt_filename = os.path.basename(text_path).replace(".txt", "")
    pdf_filename = os.path.basename(pdf_path).replace(".pdf", "")
    for idx, image in enumerate(images, start=1):
        # image = find_text_block_2(image)
        is_first_page = idx == 1
        image = preprocess_image(image,is_first_page)
        # image = layout_parser_test(image)
        # ocr_text = extract_text_from_image(image)
        ocr_data = extract_data_from_image(image)
        ocr_text = ocr_data["text"]
        # ocr_text = extract_text_from_image_easyocr(image)
        full_transcript_tmp = full_transcript
        matched_text, full_transcript, inexact, wanted_ocr_indices = extract_corresponding_transcript(ocr_text, full_transcript)
        if matched_text is None:
            print(f"Couldn't match text for pdf  {pdf_path} page {idx}.")
            continue

        img_to_save = backcrop_image(image, wanted_ocr_indices, ocr_data)

        text_alignment_match(matched_text, img_to_save)

        output_folder_tmp = f"{output_folder}inexact/" if inexact else f"{output_folder}exact/"
        Path(output_folder_tmp).mkdir(parents=True, exist_ok=True)

        with open(f"{output_folder_tmp}{txt_filename}_{idx:03}.txt", 'w') as f:
            f.write(matched_text)

        # Optionally save the image
        cv2.imwrite(f"{output_folder_tmp}{pdf_filename}_{idx:03}.png", np.array(img_to_save))


if __name__ == "__main__":
    # pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/de/year_sorted/computer/"
    pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it/pdf/"
    txt_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it/txt/"
    output_folder = "/media/fuchs/d/output_dodis_it/"
    debug = True
    single_file_pdf = "/home/fuchs/Desktop/dodis/dodo/docs_p1/pdf/26.pdf"
    single_file_txt = "/home/fuchs/Desktop/dodis/dodo/docs_p1/tessdata/26.txt"
    if not debug:
        for root, dirs, files in os.walk(pdf_path):
            for name in files:
                pdf_name = os.path.join(root, name)
                text_name = os.path.join(txt_folder, name.replace(".pdf", ".txt"))
                main(pdf_name,text_name,output_folder)
    else:
        print("Debug mode")
        pdf_name = single_file_pdf
        name = os.path.basename(single_file_pdf)
        main(pdf_name, single_file_txt, output_folder)
