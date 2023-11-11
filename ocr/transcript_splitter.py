import os
import re
import sys
from pathlib import Path

import cv2
import easyocr
import layoutparser as lp
import math
import numpy as np
import pytesseract
from PIL import ImageOps
from fuzzywuzzy import fuzz
from matplotlib import pyplot as plt, patches
from numpy import argmin, argmax
from pdf2image import convert_from_path
from pytesseract import image_to_string, image_to_data
from loguru import logger

def preprocess_pil_image(pil_image):
    # Convert PIL Image to OpenCV format (grayscale)
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    # Adaptive Thresholding
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Noise Removal with Median Filter
    # image = cv2.medianBlur(image, 3)

    # Dilation to fill in gaps
    kernel = np.ones((2,2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)

    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if 45 < angle <= 90:
        angle = angle - 90
    if not (-45 <= angle <= 45):  # The text should not be skewed more than 45 degrees.
        angle = 0

    (h, w) = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()
    return image

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

def find_matching_index_back(ocrd, transcript, start_idx_ocrd, start_idx_transcript):
    ocrd_words = ocrd[::-1]  # Split and reverse the list
    transcript_words = transcript[::-1]  # Split and reverse the list
    best_difference = float('inf')  # Set to positive infinity initially
    best_match = None
    first_match_found = False

    for j in range(0, len(ocrd_words) - 2):
        for i in range(0, len(transcript_words) - 2):
            if transcript_words[i:i + 3] == ocrd_words[j:j + 3]:
                if not first_match_found:
                    first_match_found = True
                    idx_ocrd_word = len(ocrd_words) - j - 1
                    idx_transcript_word = len(transcript_words) - i - 1
                    length_ocrd = idx_ocrd_word - start_idx_ocrd
                    length_transcript = idx_transcript_word - start_idx_transcript
                    current_difference = abs(length_ocrd - length_transcript)

                    first_match_details = (
                    idx_ocrd_word, idx_transcript_word, transcript_words[i:i + 3])
                    best_difference = current_difference



                for new_j in range(j + 1, len(ocrd_words) - 2):
                    tmp_j = new_j
                    w1 = ocrd_words[new_j]
                    w2 = ocrd_words[new_j + 1]
                    while w2 == "":
                        new_j += 1
                        w2 = ocrd_words[new_j + 1]
                    w3 = ocrd_words[new_j + 2]
                    while w3 == "":
                        new_j += 1
                        w3 = ocrd_words[new_j + 2]
                    ocrd_words_compare = [w1, w2, w3]
                    if ocrd_words_compare == transcript_words[i:i + 3]:
                        idx_ocrd_word_new = len(ocrd_words) - tmp_j - 1
                        length_ocrd_new = idx_ocrd_word_new - start_idx_ocrd
                        new_difference_ocrd = abs(length_ocrd_new - length_transcript)

                        if new_difference_ocrd < best_difference:
                            best_match = (
                                len(ocrd_words) - tmp_j - 1, len(transcript_words) - i - 1, transcript_words[i:i + 3])
                            best_difference = new_difference_ocrd

                if not best_match:
                    best_match = first_match_details

                break  # Break the inner loop after the first match is found

        if first_match_found:
            # Break the outer loop once we're done searching for a better match
            break

    if best_match:
        return best_match[0], best_match[1], False
    else:
        return -1, -1, False


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
    # print(f"Trying to load {pdf_path}")
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except:
        images = []
        # print(f"Could not load {pdf_path}. Skipping file")
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
def text_alignment_match(full_transcript, image):
    '''
    Ocr the picture again and compare the ocr with the transcript again to basically finetune and ensure that the start and end words match.
    :param full_transcript:
    :param image:
    :return:
    '''
    image =  preprocess_pil_image(image)
    ocr_data = extract_data_from_image(image)
    ocr_text = ocr_data["text"]
    # ocr_text = extract_text_from_image_easyocr(image)
    matched_text, full_transcript, inexact, wanted_ocr_indices = extract_corresponding_transcript(ocr_text,
                                                                                                  full_transcript)
    pass


def extract_corresponding_transcript(ocr_words, full_transcript):
    # ocr_words = ocr_text.split()
    margin_transcript = full_transcript.split()
    transcript_words = full_transcript.split()
    ocr_idx, transcript_idx, inexact_start = find_matching_index_front(ocr_words, full_transcript)

    if ocr_idx == -1 or transcript_idx == -1:
        return None, full_transcript, True, [], []

    ocr_words = ocr_words[ocr_idx:]
    transcript_words = transcript_words[transcript_idx:]
    margin_transcript = transcript_words[max(0,transcript_idx-10):]
    ocr_words = remove_dodis(ocr_words)

    ocr_end_index, transcript_end_idx, inexact_end = find_matching_index_back(ocr_words, transcript_words, ocr_idx, transcript_idx)

    relevant_indices = list(range(ocr_idx, ocr_end_index+ocr_idx + 1))

    ocr_words_no_space = ocr_words[:ocr_end_index + 1]
    ocr_words_no_space = [word for word in ocr_words_no_space if word != ""]
    transcript_matched_words = transcript_words[:transcript_end_idx+1]
    if abs(len(ocr_words_no_space)-(len(transcript_matched_words))) > 20:
        logger.warning(f"We have a huge diffeernce between the ocr adn transcript end indices. Skipping. "
                       f"OCR idx diff {len(ocr_words_no_space)} and transcript diff {len(transcript_matched_words)}")
        return None, full_transcript, True, [], []
    if ocr_end_index == -1 or transcript_end_idx == -1 or len(relevant_indices) < 15 or len(transcript_words) < 15:
        return None, full_transcript, True, [], []

    # Extract the corresponding text from the transcript
    matched_text = ' '.join(transcript_words[:transcript_end_idx+1])
    remaining_transcript = ' '.join(transcript_words[transcript_end_idx+1:])
    margin_transcript_text = " ".join(margin_transcript[:min(transcript_end_idx+10,len(margin_transcript))])
    inexact = inexact_start or inexact_end

    return matched_text, remaining_transcript, inexact, relevant_indices, margin_transcript_text


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
def are_values_close(lst, margin):
    return max(lst) - min(lst) <= margin


def is_three_word_match(i, transcript_list, first_words_start):
    for j in range(len(first_words_start)):
        if i + j >= len(transcript_list):
            return False
        transcript_word = transcript_list[i + j]
        if fuzz.ratio(first_words_start[j], transcript_word) < 80:
            return False
    return True


def backcrop_image(image, indices_to_keep, tessdata, matched_text, take_out_parts = False):
    plots = False
    transcript_list = matched_text.split()
    if not transcript_list:
        return image, None
    word_boxes = {index:(
        tessdata["left"][index],
        tessdata["top"][index],
        tessdata["left"][index] + tessdata["width"][index],
        tessdata["top"][index] + tessdata["height"][index]
    ) for index in indices_to_keep}

    # TODO: Factor in one liner cases. Hoiw to?
    # Mayeb check if heights of all boxes are within 5 of eachother. then its one lines
    # if it is, alxo cut off x.  For now skip 1 liners
    heights = [(box[3] - box[1]) for box in word_boxes.values()]


    average_height = sum(heights) / len(heights)

    word_boxes = {idx:box for idx,box in word_boxes.items() if (box[3] - box[1]) <= 1.7 * average_height}
    # remove boxes withe empty amtches
    word_boxes = {idx:box for idx,box in word_boxes.items() if tessdata["text"][idx] != ""}

    # remove empty indices to keep
    indices_to_keep = [idx for idx in indices_to_keep if tessdata["text"][idx] != ""]

    y_1_s = [box[1] for box in word_boxes.values()]

    if are_values_close(y_1_s, 1.3 * average_height) or len(indices_to_keep) < 15 or len(transcript_list) < 15:
        logger.warning("Skipping backcrop, as it seems to be a one liner")
        return None, None

    crop_x1 = min([box[0] for box in word_boxes.values()])
    crop_y1 = min([box[1] for box in word_boxes.values()])
    crop_x2 = max([box[2] for box in word_boxes.values()])
    crop_y2 = max([box[3] for box in word_boxes.values()])

    # Copy the original image for manipulation
    masked_image = image.copy()
    cropi_boxi_first = None
    cropi_boxi_last = None
    # Check boxes that are still on the image and crop bot and top there, and ismply remove them. TRhen also reomve recognized words that goit cropedp off now from
    # the final transcript.
    # Somehow we need to check for "random" boxes, thta are in the middle of the image
    # For outliers add x back in
    for index in set(range(len(tessdata["left"]))) - set(indices_to_keep):
        x1, y1, x2, y2 = (
            tessdata["left"][index],
            tessdata["top"][index],
            tessdata["left"][index] + tessdata["width"][index],
            tessdata["top"][index] + tessdata["height"][index]
        )
        # Only check bottom of boxes for top crop and top of box for bottom crop, to prevent
        # half words
        if tessdata["text"][index] == "":
            continue

        if crop_x1 -10 <= x1 <= crop_x2+10 and crop_x1-10 <= x2 <= crop_x2+10:
            if index < indices_to_keep[0] and crop_y1 <= y2 <= crop_y2:
                # and (crop_y1 <= y1 <= crop_y2 and crop_y1 <= y2 <= crop_y2 and crop_x1 -10 <= x1 <= crop_x2+10 and crop_x1-10 <= x2 <= crop_x2+10):
                diff_top = abs(y1 - crop_y1)
                if diff_top > 4 * average_height:
                    logger.warning(f"diff top seems unreasonable, skip box {index}, diff is {diff_top}")
                    continue
                crop_y1 = y2
                cropi_boxi_first = (x1,y1,x2,y2)
            elif index > indices_to_keep[-1] and crop_y1 <= y1 <= crop_y2:
                diff_bot = abs(crop_y2 - y2)
                if diff_bot > 4 * average_height:
                    logger.warning(f"diff bot seems unreasonable, skip box {index}, diff is {diff_bot}")
                    continue
                crop_y2 = y1
                cropi_boxi_last = (x1,y1,x2,y2)


    dropped_off_indices = []
    cntr = 0
    # from start
    first_words_start = []
    for index in word_boxes.keys():
        x1, y1, x2, y2 = (
            tessdata["left"][index],
            tessdata["top"][index],
            tessdata["left"][index] + tessdata["width"][index],
            tessdata["top"][index] + tessdata["height"][index]
        )
        #if not (crop_y1 <= y1 <= crop_y2) and not crop_y1 <= y2 <= crop_y2 \
        #        and not crop_y1 <= y1 + average_height <= crop_y2:
        if tessdata["text"][index] == "":
            continue
        if y1 < crop_y1 - 0.2 * average_height:
            dropped_off_indices.append(index)
            cntr = 0
            first_words_start = []
        else:
            cntr += 1
            if not first_words_start:
                tmp_idi = index
                while len(first_words_start) < 3:
                    word = tessdata["text"][tmp_idi]
                    if word != "":
                        first_words_start.append(word)
                    tmp_idi += 1
        if cntr == 3:
            break
    #from end
    cntr = 0
    first_words_end = []
    for index in list(word_boxes.keys())[::-1]:
        x1, y1, x2, y2 = (
            tessdata["left"][index],
            tessdata["top"][index],
            tessdata["left"][index] + tessdata["width"][index],
            tessdata["top"][index] + tessdata["height"][index]
        )
        if tessdata["text"][index] == "":
            continue
        if y2 > crop_y2 + 0.2 * average_height:
            dropped_off_indices.append(index)
            cntr = 0
            first_words_end = []
        else:
            cntr += 1
            if not first_words_end:
                tmp_idi = index
                while len(first_words_end) < 3:
                    word = tessdata["text"][tmp_idi]
                    if word != "":
                        first_words_end.append(word)
                    tmp_idi -= 1
        if cntr == 3:
            break

    dropped_off_indices = sorted(dropped_off_indices)

    transcript_indices_to_rm_start = []
    three_word_match_start = False
    # First try to reomve from start by looking at a enaxct match for the first three
    #words on the new line.
    logger.trace(f"Attempt to remove exactly by first three words. The words are: {first_words_start}")
    if dropped_off_indices and dropped_off_indices[0] == indices_to_keep[0]:
        for i, word in enumerate(transcript_list):
            if not is_three_word_match(i,transcript_list,first_words_start):
                transcript_indices_to_rm_start.append(i)
            else:
                three_word_match_start = True
                logger.success(f"Found a three word match for the first three words. The words are: {[transcript_list[i+j] for j in range(3)]}")
                break

        if three_word_match_start:
            transcript_list = [ele for idx, ele in enumerate(transcript_list) if
                                   idx not in transcript_indices_to_rm_start]

    rm_from_start = 0
    current_diff_start = 0
    rm_from_end = 0
    current_diff_end = 0
    if not three_word_match_start and dropped_off_indices and dropped_off_indices[0] == indices_to_keep[0]:
        logger.warning(f"No exact match for first three words on line, attmept approx")
        for i,idx in enumerate(dropped_off_indices):
            if idx in word_boxes.keys():
                del word_boxes[idx]
            if idx == indices_to_keep[i]:
                rm_from_start += 1
                # Add check for things like z. B. where there is a spüacei in ebtwen
                transcript_word = transcript_list[i]
                ocr_word = tessdata["text"][idx]
                if ocr_word != transcript_word and ocr_word.startswith(transcript_word) and i+1 < len(transcript_list):
                    next_transcript_word = transcript_list[i+1]
                    combined_word = transcript_word + next_transcript_word
                    if combined_word == ocr_word:
                        logger.warning(
                            f"Found off behavior in transcript, where the word {ocr_word} starts with {transcript_word}")
                        logger.warning(f"Attempt to resovle, as the combined word {combined_word} matches the ocr word {ocr_word}")
                        rm_from_start += 1

    transcript_indices_to_rm_end = []
    three_word_match_end = False
    # First try to reomve from start by looking at a enaxct match for the first three
    # words on the new line.
    logger.trace(f"Attempt to remove exactly by last three words. The words are: {first_words_end}")
    if dropped_off_indices and dropped_off_indices[-1] == indices_to_keep[-1]:
        for i, word in enumerate(transcript_list[::-1]):
            if not is_three_word_match(i, transcript_list[::-1], first_words_end):
                transcript_indices_to_rm_end.append(i)
            else:
                three_word_match_end = True
                logger.success(
                    f"Found a three word match for the last three words. The words are: {[transcript_list[-(i + j+1)] for j in range(3)]}")
                break

        if three_word_match_end:
            transcript_list_rev = [ele for idx, ele in enumerate(transcript_list[::-1]) if idx not in transcript_indices_to_rm_end]
            transcript_list = transcript_list_rev[::-1]
    if not three_word_match_end and dropped_off_indices and dropped_off_indices[-1] == indices_to_keep[-1]:
        logger.warning(f"No exact match for last three words on line, attmept approx")
        for i, back_idx in enumerate(dropped_off_indices[::-1]):
            if back_idx in word_boxes.keys():
                del word_boxes[back_idx]
            if back_idx == indices_to_keep[::-1][i]:
                rm_from_end += 1
                # Add check for things like z. B. where there is a spüacei in ebtwen
                transcript_word = transcript_list[-i]
                ocr_word = tessdata["text"][back_idx]
                if ocr_word != transcript_word and ocr_word.endswith(transcript_word) and i + 1 < len(transcript_list):
                    next_transcript_word = transcript_list[-(i + 1)]
                    combined_word = next_transcript_word + transcript_word
                    if combined_word == ocr_word:
                        logger.warning(f"Found off behavior in transcript, where the word {ocr_word} ends with {transcript_word}")
                        logger.warning(f"Attempt to resovle, as the combined word {combined_word} matches the ocr word {ocr_word}")
                        rm_from_end += 1



    for i in range(rm_from_start):
        if not transcript_list or len(transcript_list) < 15:
            logger.warning(f"Abort. The transcript list is empty or too short to remove from start. The transcript list is: {transcript_list}")
            return None, None
        word_to_be_removed = re.sub('[^a-zA-Z]+', '', transcript_list[0])
        # first_word_start_line_start = first_words_start[0].replace("-","")
        first_word_start_line_start = re.sub('[^a-zA-Z]+', '', first_words_start[0])
        if word_to_be_removed != "" and first_word_start_line_start != "" and word_to_be_removed.endswith(first_word_start_line_start):
            logger.warning(f"Stopped removing, as it seems a - case: {first_word_start_line_start}, {transcript_list[0]}")
            transcript_list[0] = first_words_start[0]
            break

        fuzz_ratio = fuzz.ratio(first_words_start[0],word_to_be_removed)
        if word_to_be_removed != "" and fuzz_ratio >=75:
            logger.warning(f"Did not remove from start, as we foudn a match with SED to the presumed start fo the line word. The words compared are: {first_words_start[0]}, {transcript_list[0]}, Ratio: {fuzz_ratio}")
            break

        if fuzz_ratio < 75:
            del transcript_list[0]

    for i in range(rm_from_end):
        if not transcript_list or len(transcript_list) < 15:
            logger.warning(f"Abort. The transcript list is empty or too short to remove from end. The transcript list is: {transcript_list}")
            return None, None
        word_to_be_removed = re.sub('[^a-zA-Z]+', '', transcript_list[-1])
        first_word_end_line_end = re.sub('[^a-zA-Z]+', '', first_words_end[0])
        # first_word_end_line_end = first_words_end[0].replace("-","")
        if word_to_be_removed != "" and first_word_end_line_end != "" and word_to_be_removed.startswith(first_word_end_line_end):
            logger.warning(f"Stopped removing, as it seems a - case: {first_word_end_line_end}, {transcript_list[-1]}")
            transcript_list[-1] = first_words_end[0]
            break

        fuzz_ratio = fuzz.ratio(first_words_end[0],word_to_be_removed)
        if word_to_be_removed != "" and fuzz_ratio >=75:
            logger.warning(
                f"Did not remove from end, as we foudn a match with SED to the presumed start fo the line word. The words compared are: {first_words_end[0]}, {transcript_list[-1]}, Ratio: {fuzz_ratio}")
            break

        if fuzz_ratio < 75:
            del transcript_list[-1]
    # Add padding
    padding = 2
    crop_x1 = 0
    # crop_x1 = max(0, crop_x1 - padding)
    crop_y1 = max(0, crop_y1 - 2*padding)
    crop_x2 = image.shape[1]
    # crop_x2 = min(image.shape[1], crop_x2 + padding)
    crop_y2 = min(image.shape[0], crop_y2 + padding)

    # Crop the image
    cropped_masked_image = masked_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if plots:
        idx_x2 = argmax([box[2] for box in word_boxes.values()])
        idx_y2 = argmax([box[3] for box in word_boxes.values()])

        # Plot the original image with bounding boxes
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
        # for box in list(word_boxes.values())[-10:]:
        #     rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)

        # box_max_x = list(word_boxes.values())[idx_x2]
        # rect_max_x = patches.Rectangle((box_max_x[0], box_max_x[1]), box_max_x[2] - box_max_x[0], box_max_x[3] - box_max_x[1], linewidth=1, edgecolor='g', facecolor='none')
        # ax.add_patch(rect_max_x)

        # box_max_y = list(word_boxes.values())[idx_y2]
        # rect_max_y = patches.Rectangle((box_max_y[0], box_max_y[1]), box_max_y[2] - box_max_y[0], box_max_y[3] - box_max_y[1], linewidth=1, edgecolor='g', facecolor='none')
        # ax.add_patch(rect_max_y)

        if cropi_boxi_first is not None:
            rect_min_y = patches.Rectangle((cropi_boxi_first[0], cropi_boxi_first[1]), cropi_boxi_first[2] - cropi_boxi_first[0]+20,
                                           cropi_boxi_first[3] - cropi_boxi_first[1]+20, linewidth=3, edgecolor='g', facecolor='none')

            ax.add_patch(rect_min_y)
        if cropi_boxi_last is not None:
            rect_max_y = patches.Rectangle((cropi_boxi_last[0], cropi_boxi_last[1]), cropi_boxi_last[2] - cropi_boxi_last[0]+20,
                                           cropi_boxi_last[3] - cropi_boxi_last[1]+20, linewidth=3, edgecolor='r', facecolor='none')
            ax.add_patch(rect_max_y)
        # idi = 18
        # fancy_box = (
        # tessdata["left"][idi],
        # tessdata["top"][idi],
        # tessdata["left"][idi] + tessdata["width"][idi],
        # tessdata["top"][idi] + tessdata["height"][idi]
        # )
        # fancy_box_rect = patches.Rectangle((fancy_box[0], fancy_box[1]), fancy_box[2] - fancy_box[0],
        #                                fancy_box[3] - fancy_box[1], linewidth=1, edgecolor='y', facecolor='none')
        # ax.add_patch(fancy_box_rect)

        plt.title("Original Image with Bounding Boxes")
        plt.axis('off')
        plt.show()
        plt.imshow(cropped_masked_image)
        plt.axis('off')
        plt.show()

    return cropped_masked_image, " ".join(transcript_list)

def plot_n_boxes(image, boxes, n):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for box in boxes[-n:]:
        rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.title("Original Image with Bounding Boxes")
    plt.axis('off')
    plt.show()

def footnotes_idx_in_main(wanted_ocr_indices_fn, wanted_ocr_indices):
    for idx in wanted_ocr_indices_fn:
        if idx in wanted_ocr_indices:
            return True
    return False


def main(pdf_path,text_path,footnotes_text_path,output_folder):
    with open(text_path, 'r') as f:
        full_transcript = f.read()
    with open(footnotes_text_path, 'r') as f:
        full_footnotes_transcript = f.read()

    images = convert_pdf_to_images(pdf_path)
    # images = convert_from_path(pdf_path, dpi=300)
    # images = [auto_crop(img) for img in images]

    txt_filename = os.path.basename(text_path).replace(".txt", "")
    pdf_filename = os.path.basename(pdf_path).replace(".pdf", "")
    logger.trace(f"Processing {pdf_filename}")
    for idx, image in enumerate(images, start=1):
        logger.trace(f"Processing page {idx} of {len(images)}")
        # image = find_text_block_2(image)
        # if idx != 5:
        #     continue
        is_first_page = idx == 1
        image = preprocess_image(image,is_first_page)
        orig_img = image.copy()
        image = preprocess_pil_image(image)
        # image = layout_parser_test(image)
        # ocr_text = extract_text_from_image(image)
        ocr_data = extract_data_from_image(image)
        ocr_text = ocr_data["text"]
        # ocr_text = extract_text_from_image_easyocr(image)
        full_transcript_tmp = full_transcript
        logger.trace(f"Trying to match main text for pdf  {pdf_path} page {idx}.")
        # Main etxt --------------------
        matched_text, full_transcript, inexact, wanted_ocr_indices, margin_transcript =  extract_corresponding_transcript(ocr_text, full_transcript)
        # Footnotes --------------------
        matched_footnotes, _, inexact_fn, wanted_ocr_indices_fn, margin_transcript_fn = extract_corresponding_transcript(
            ocr_text, full_footnotes_transcript)

        if matched_text is None or len(matched_text.split()) == 0:
            logger.warning(f"Couldn't match main text for pdf  {pdf_path} page {idx}.")
        else:
            logger.success(f"Matched main text for pdf  {pdf_path} page {idx}.")
            logger.trace(f"Trying to backcrop image for pdf  {pdf_path} page {idx}.")
            img_to_save, matched_text = backcrop_image(orig_img, wanted_ocr_indices, ocr_data, matched_text)

             # text_alignment_match(margin_transcript, img_to_save)

            output_folder_tmp = f"{output_folder}inexact/" if inexact else f"{output_folder}exact/"
            Path(output_folder_tmp).mkdir(parents=True, exist_ok=True)

            if img_to_save is not None:
                with open(f"{output_folder_tmp}{txt_filename}_{idx:03}.txt", 'w') as f:
                    f.write(matched_text)

                # Optionally save the image
                cv2.imwrite(f"{output_folder_tmp}{pdf_filename}_{idx:03}.png", np.array(img_to_save))
                logger.success(f"Saved page {idx} of {len(images)}")
                logger.success(f"Final words in transcript are {len(matched_text.split())}")
        if matched_footnotes is None or len(matched_footnotes.split()) == 0 or footnotes_idx_in_main(wanted_ocr_indices_fn, wanted_ocr_indices):
            logger.warning(f"Couldn't match footnotes for pdf  {pdf_path} page {idx}.")
        else:
            logger.success(f"Matched footnotes for pdf  {pdf_path} page {idx}.")
            logger.trace(f"Trying to backcrop footnote image for pdf  {pdf_path} page {idx}.")
            img_to_save, matched_footnotes = backcrop_image(orig_img, wanted_ocr_indices_fn, ocr_data, matched_footnotes)
            output_folder_tmp = f"{output_folder}inexact/" if inexact_fn else f"{output_folder}exact/"
            Path(output_folder_tmp).mkdir(parents=True, exist_ok=True)

            if img_to_save is not None:
                with open(f"{output_folder_tmp}{txt_filename}_{idx:03}_footnotes.txt", 'w') as f:
                    f.write(matched_footnotes)
                # Optionally save the image
                cv2.imwrite(f"{output_folder_tmp}{pdf_filename}_{idx:03}_footnotes.png", np.array(img_to_save))
                logger.success(f"Saved fontoote of page {idx} of {len(images)}")
                logger.success(f"Final words in transcript are {len(matched_footnotes.split())}")

if __name__ == "__main__":
    pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/fr/year_sorted/computer/part1/"
    # pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it/year_sorted/computer/"
    # pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/en/computer/"
    txt_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/text_transcripts/"
    output_folder = "/media/fuchs/d/dataset_try_4/parts/fr/"
    # output_folder = "/media/fuchs/d/testi_output/"
    log_file = f"{output_folder}creatino_de_log_1.txt"

    debug = False
    dbeug_pdf_base = "/home/fuchs/Desktop/dodis/dodo/docs_p1/pdf/"
    pdf_list = [23,26,3,42968,55703,30751,45823,55703,48366,47372,54174,32700,8303,54813, 35754]
    pdf_list = [45823,55703,48366,47372,54174,32700,8303,54813, 35754]
    pdf_list = [54174]
    # pdf_list = [10156]
    debug_txt_base = "/home/fuchs/Desktop/dodis/dodo/docs_p1/text_transcripts/"
    debug_txt_fn = "/home/fuchs/Desktop/dodis/dodo/docs_p1/text_transcripts/"

    debug_pdf_fnames = [f"{dbeug_pdf_base}{pdf_name}.pdf" for pdf_name in pdf_list]
    debug_txt_fnamse = [f"{debug_txt_base}{pdf_name}.txt" for pdf_name in pdf_list]
    debug_txt_fn_fnamse = [f"{debug_txt_fn}footnotes_{pdf_name}.txt" for pdf_name in pdf_list]

    log_level = "TRACE"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(sys.stdout, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add(log_file, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)

    #23, 26,3, 42968
    #35754_011
    #55703_004_footnotes
    #30751_005 footnoretsa
    # 45823_008 footnotes
    #55703_005_foothn
    # 48366_004
    #47372_001
    #54174_003
    #327000_001
    #8303_001
    #54813_008 footntoes
    if not debug:
        for root, dirs, files in os.walk(pdf_path):
            for name in files:
                pdf_name = os.path.join(root, name)
                text_name = os.path.join(txt_folder, name.replace(".pdf", ".txt"))
                footnote_text_name = os.path.join(txt_folder, f"footnotes_{name}".replace(".pdf", ".txt"))
                main(pdf_name,text_name,footnote_text_name, output_folder)
    else:
        print("Debug mode")
        for pdf_name, single_file_txt, single_file_txt_fn in zip(debug_pdf_fnames, debug_txt_fnamse, debug_txt_fn_fnamse):
            main(pdf_name, single_file_txt,single_file_txt_fn, output_folder)
