import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import ImageOps
from fuzzywuzzy import fuzz
from loguru import logger
from pdf2image import convert_from_path
from pytesseract import image_to_data, TesseractError


def preprocess_pil_image(pil_image):
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)

    kernel = np.ones((2, 2), np.uint8)
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

    return -1, -1, inexact  # Return -1 if no match found


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
        return len(ocrd_words) - j - 1, len(
            transcript_words) - index - 1, inexact  # Convert index back to original order

    return -1, -1, inexact


def convert_pdf_to_images(pdf_path):
    # print(f"Trying to load {pdf_path}")
    try:
        images = convert_from_path(pdf_path, dpi=300)
    except:
        images = []
        # print(f"Could not load {pdf_path}. Skipping file")
    return images


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


def extract_data_from_image(image):
    "deu+fra+ita+eng"
    try:
        return image_to_data(image, output_type=pytesseract.Output.DICT, lang="deu+fra+ita+eng")
    except TesseractError:
        return image_to_data(image, output_type=pytesseract.Output.DICT)


def remove_dodis(ocr_words):
    for i in range(len(ocr_words)):
        if ocr_words[i].lower() == "dodis":
            return ocr_words[:i]
    return ocr_words


def extract_corresponding_transcript(ocr_words, full_transcript):
    # ocr_words = ocr_text.split()
    margin_transcript = full_transcript.split()
    transcript_words = full_transcript.split()
    ocr_idx, transcript_idx, inexact_start = find_matching_index_front(ocr_words, full_transcript)

    if ocr_idx == -1 or transcript_idx == -1:
        return None, full_transcript, True, [], []

    ocr_words = ocr_words[ocr_idx:]
    transcript_words = transcript_words[transcript_idx:]
    margin_transcript = transcript_words[max(0, transcript_idx - 10):]
    ocr_words = remove_dodis(ocr_words)

    ocr_end_index, transcript_end_idx, inexact_end = find_matching_index_back(ocr_words, transcript_words, ocr_idx,
                                                                              transcript_idx)

    relevant_indices = list(range(ocr_idx, ocr_end_index + ocr_idx + 1))

    ocr_words_no_space = ocr_words[:ocr_end_index + 1]
    ocr_words_no_space = [word for word in ocr_words_no_space if word != ""]
    transcript_matched_words = transcript_words[:transcript_end_idx + 1]
    if abs(len(ocr_words_no_space) - (len(transcript_matched_words))) > 20:
        logger.warning(f"We have a huge diffeernce between the ocr adn transcript end indices. Skipping. "
                       f"OCR idx diff {len(ocr_words_no_space)} and transcript diff {len(transcript_matched_words)}")
        return None, full_transcript, True, [], []
    if ocr_end_index == -1 or transcript_end_idx == -1 or len(relevant_indices) < 15 or len(transcript_words) < 15:
        return None, full_transcript, True, [], []

    # Extract the corresponding text from the transcript
    matched_text = ' '.join(transcript_words[:transcript_end_idx + 1])
    remaining_transcript = ' '.join(transcript_words[transcript_end_idx + 1:])
    margin_transcript_text = " ".join(margin_transcript[:min(transcript_end_idx + 10, len(margin_transcript))])
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

    img_cropped = np.array(image)[start_y:, :]
    return img_cropped



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


def backcrop_image(image, indices_to_keep, tessdata, matched_text, take_out_parts=False):
    plots = False
    transcript_list = matched_text.split()
    if not transcript_list:
        return image, None
    word_boxes = {index: (
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

    word_boxes = {idx: box for idx, box in word_boxes.items() if (box[3] - box[1]) <= 1.7 * average_height}
    # remove boxes withe empty amtches
    word_boxes = {idx: box for idx, box in word_boxes.items() if tessdata["text"][idx] != ""}

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

    for index in set(range(len(tessdata["left"]))) - set(indices_to_keep):
        x1, y1, x2, y2 = (
            tessdata["left"][index],
            tessdata["top"][index],
            tessdata["left"][index] + tessdata["width"][index],
            tessdata["top"][index] + tessdata["height"][index]
        )
        if tessdata["text"][index] == "":
            continue

        if crop_x1 - 10 <= x1 <= crop_x2 + 10 and crop_x1 - 10 <= x2 <= crop_x2 + 10:
            if index < indices_to_keep[0] and crop_y1 <= y2 <= crop_y2:
                # and (crop_y1 <= y1 <= crop_y2 and crop_y1 <= y2 <= crop_y2 and crop_x1 -10 <= x1 <= crop_x2+10 and crop_x1-10 <= x2 <= crop_x2+10):
                diff_top = abs(y1 - crop_y1)
                if diff_top > 4 * average_height:
                    logger.warning(f"diff top seems unreasonable, skip box {index}, diff is {diff_top}")
                    continue
                crop_y1 = y2
                cropi_boxi_first = (x1, y1, x2, y2)
            elif index > indices_to_keep[-1] and crop_y1 <= y1 <= crop_y2:
                diff_bot = abs(crop_y2 - y2)
                if diff_bot > 4 * average_height:
                    logger.warning(f"diff bot seems unreasonable, skip box {index}, diff is {diff_bot}")
                    continue
                crop_y2 = y1
                cropi_boxi_last = (x1, y1, x2, y2)

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
        # if not (crop_y1 <= y1 <= crop_y2) and not crop_y1 <= y2 <= crop_y2 \
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
    # from end
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
    # words on the new line.
    logger.trace(f"Attempt to remove exactly by first three words. The words are: {first_words_start}")
    if dropped_off_indices and dropped_off_indices[0] == indices_to_keep[0]:
        for i, word in enumerate(transcript_list):
            if not is_three_word_match(i, transcript_list, first_words_start):
                transcript_indices_to_rm_start.append(i)
            else:
                three_word_match_start = True
                logger.success(
                    f"Found a three word match for the first three words. The words are: {[transcript_list[i + j] for j in range(3)]}")
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
        for i, idx in enumerate(dropped_off_indices):
            if idx in word_boxes.keys():
                del word_boxes[idx]
            if idx == indices_to_keep[i]:
                rm_from_start += 1
                # Add check for things like z. B. where there is a spüacei in ebtwen
                transcript_word = transcript_list[i]
                ocr_word = tessdata["text"][idx]
                if ocr_word != transcript_word and ocr_word.startswith(transcript_word) and i + 1 < len(
                        transcript_list):
                    next_transcript_word = transcript_list[i + 1]
                    combined_word = transcript_word + next_transcript_word
                    if combined_word == ocr_word:
                        logger.warning(
                            f"Found off behavior in transcript, where the word {ocr_word} starts with {transcript_word}")
                        logger.warning(
                            f"Attempt to resovle, as the combined word {combined_word} matches the ocr word {ocr_word}")
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
                    f"Found a three word match for the last three words. The words are: {[transcript_list[-(i + j + 1)] for j in range(3)]}")
                break

        if three_word_match_end:
            transcript_list_rev = [ele for idx, ele in enumerate(transcript_list[::-1]) if
                                   idx not in transcript_indices_to_rm_end]
            transcript_list = transcript_list_rev[::-1]
    if not three_word_match_end and dropped_off_indices and dropped_off_indices[-1] == indices_to_keep[-1]:
        logger.warning(f"No exact match for last three words on line, attmept approx")
        for i, back_idx in enumerate(dropped_off_indices[::-1]):
            if back_idx in word_boxes.keys():
                del word_boxes[back_idx]
            if back_idx == indices_to_keep[::-1][i]:
                rm_from_end += 1
                # Add check for things like z. B. where there is a spüacei in ebtwen
                if i < len(transcript_list):
                    transcript_word = transcript_list[-i]
                    ocr_word = tessdata["text"][back_idx]
                    if ocr_word != transcript_word and ocr_word.endswith(transcript_word) and i + 1 < len(
                            transcript_list):
                        next_transcript_word = transcript_list[-(i + 1)]
                        combined_word = next_transcript_word + transcript_word
                        if combined_word == ocr_word:
                            logger.warning(
                                f"Found off behavior in transcript, where the word {ocr_word} ends with {transcript_word}")
                            logger.warning(
                                f"Attempt to resovle, as the combined word {combined_word} matches the ocr word {ocr_word}")
                            rm_from_end += 1

    for i in range(rm_from_start):
        if not transcript_list or len(transcript_list) < 15:
            logger.warning(
                f"Abort. The transcript list is empty or too short to remove from start. The transcript list is: {transcript_list}")
            return None, None
        word_to_be_removed = re.sub('[^a-zA-Z]+', '', transcript_list[0])
        # first_word_start_line_start = first_words_start[0].replace("-","")
        first_word_start_line_start = re.sub('[^a-zA-Z]+', '', first_words_start[0])
        if word_to_be_removed != "" and first_word_start_line_start != "" and word_to_be_removed.endswith(
                first_word_start_line_start):
            logger.warning(
                f"Stopped removing, as it seems a - case: {first_word_start_line_start}, {transcript_list[0]}")
            transcript_list[0] = first_words_start[0]
            break

        fuzz_ratio = fuzz.ratio(first_words_start[0], word_to_be_removed)
        if word_to_be_removed != "" and fuzz_ratio >= 75:
            logger.warning(
                f"Did not remove from start, as we foudn a match with SED to the presumed start fo the line word. The words compared are: {first_words_start[0]}, {transcript_list[0]}, Ratio: {fuzz_ratio}")
            break

        if fuzz_ratio < 75:
            del transcript_list[0]

    for i in range(rm_from_end):
        if not transcript_list or len(transcript_list) < 15:
            logger.warning(
                f"Abort. The transcript list is empty or too short to remove from end. The transcript list is: {transcript_list}")
            return None, None
        word_to_be_removed = re.sub('[^a-zA-Z]+', '', transcript_list[-1])
        first_word_end_line_end = re.sub('[^a-zA-Z]+', '', first_words_end[0])
        # first_word_end_line_end = first_words_end[0].replace("-","")
        if word_to_be_removed != "" and first_word_end_line_end != "" and word_to_be_removed.startswith(
                first_word_end_line_end):
            logger.warning(f"Stopped removing, as it seems a - case: {first_word_end_line_end}, {transcript_list[-1]}")
            transcript_list[-1] = first_words_end[0]
            break

        fuzz_ratio = fuzz.ratio(first_words_end[0], word_to_be_removed)
        if word_to_be_removed != "" and fuzz_ratio >= 75:
            logger.warning(
                f"Did not remove from end, as we foudn a match with SED to the presumed start fo the line word. The words compared are: {first_words_end[0]}, {transcript_list[-1]}, Ratio: {fuzz_ratio}")
            break

        if fuzz_ratio < 75:
            del transcript_list[-1]
    # Add padding
    padding = 2
    crop_x1 = 0
    # crop_x1 = max(0, crop_x1 - padding)
    crop_y1 = max(0, crop_y1 - 2 * padding)
    crop_x2 = image.shape[1]
    # crop_x2 = min(image.shape[1], crop_x2 + padding)
    crop_y2 = min(image.shape[0], crop_y2 + padding)

    # Crop the image
    cropped_masked_image = masked_image[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_masked_image, " ".join(transcript_list)


def footnotes_idx_in_main(wanted_ocr_indices_fn, wanted_ocr_indices):
    for idx in wanted_ocr_indices_fn:
        if idx in wanted_ocr_indices:
            return True
    return False


def main(pdf_path, text_path, footnotes_text_path, output_folder):
    with open(text_path, 'r') as f:
        full_transcript = f.read()
    with open(footnotes_text_path, 'r') as f:
        full_footnotes_transcript = f.read()

    images = convert_pdf_to_images(pdf_path)

    txt_filename = os.path.basename(text_path).replace(".txt", "")
    pdf_filename = os.path.basename(pdf_path).replace(".pdf", "")
    logger.trace(f"Processing {pdf_filename}")
    for idx, image in enumerate(images, start=1):
        logger.trace(f"Processing page {idx} of {len(images)}")
        # image = find_text_block_2(image)
        # if idx != 3:
        #     continue
        is_first_page = idx == 1
        image = remove_white_border(image)
        if is_first_page:
            image = remove_top_part(image)
        image = np.array(image)
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
        matched_text, full_transcript, inexact, wanted_ocr_indices, margin_transcript = extract_corresponding_transcript(
            ocr_text, full_transcript)
        # Footnotes --------------------
        matched_footnotes, _, inexact_fn, wanted_ocr_indices_fn, margin_transcript_fn = extract_corresponding_transcript(
            ocr_text, full_footnotes_transcript)

        if matched_text is None or len(matched_text.split()) == 0:
            logger.warning(f"Couldn't match main text for pdf  {pdf_path} page {idx}.")
        else:
            logger.success(f"Matched main text for pdf  {pdf_path} page {idx}.")
            logger.trace(f"Trying to backcrop image for pdf  {pdf_path} page {idx}.")
            try:
                img_to_save, matched_text = backcrop_image(orig_img, wanted_ocr_indices, ocr_data, matched_text)
                output_folder_tmp = f"{output_folder}inexact/" if inexact else f"{output_folder}exact/"
                Path(output_folder_tmp).mkdir(parents=True, exist_ok=True)

            except:
                logger.warning(f"Something went wrong with backcropping. Skipping")
                img_to_save = None

            if img_to_save is not None:
                with open(f"{output_folder_tmp}{txt_filename}_{idx:03}.txt", 'w') as f:
                    f.write(matched_text)

                # Optionally save the image
                cv2.imwrite(f"{output_folder_tmp}{pdf_filename}_{idx:03}.png", np.array(img_to_save))
                logger.success(f"Saved page {idx} of {len(images)}")
                logger.success(f"Final words in transcript are {len(matched_text.split())}")
        if matched_footnotes is None or len(matched_footnotes.split()) == 0 or footnotes_idx_in_main(
                wanted_ocr_indices_fn, wanted_ocr_indices):
            logger.warning(f"Couldn't match footnotes for pdf  {pdf_path} page {idx}.")
        else:
            logger.success(f"Matched footnotes for pdf  {pdf_path} page {idx}.")
            logger.trace(f"Trying to backcrop footnote image for pdf  {pdf_path} page {idx}.")
            try:
                img_to_save, matched_footnotes = backcrop_image(orig_img, wanted_ocr_indices_fn, ocr_data,
                                                                matched_footnotes)
                output_folder_tmp = f"{output_folder}inexact/" if inexact_fn else f"{output_folder}exact/"
                Path(output_folder_tmp).mkdir(parents=True, exist_ok=True)
            except:
                logger.warning(f"Something went wrong with backcropping. Skipping")
                img_to_save = None
            if img_to_save is not None:
                with open(f"{output_folder_tmp}{txt_filename}_{idx:03}_footnotes.txt", 'w') as f:
                    f.write(matched_footnotes)
                # Optionally save the image
                cv2.imwrite(f"{output_folder_tmp}{pdf_filename}_{idx:03}_footnotes.png", np.array(img_to_save))
                logger.success(f"Saved fontoote of page {idx} of {len(images)}")
                logger.success(f"Final words in transcript are {len(matched_footnotes.split())}")


if __name__ == "__main__":
    pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/fr/year_sorted/computer/part3/"
    # pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/it/year_sorted/computer/"
    # pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/en/computer/"
    txt_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/text_transcripts/"
    output_folder = "/media/fuchs/d/dataset_try_6/parts/fr/"
    # output_folder = "/media/fuchs/d/testi_output/"
    log_file = f"{output_folder}creatino_de_log_3.txt"

    debug = False
    dbeug_pdf_base = "/home/fuchs/Desktop/dodis/dodo/docs_p1/pdf/"
    pdf_list = [23, 26, 3, 42968, 55703, 30751, 45823, 55703, 48366, 47372, 54174, 32700, 8303, 54813, 35754]
    pdf_list = [45823, 55703, 48366, 47372, 54174, 32700, 8303, 54813, 35754]
    pdf_list = [39653]
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

    # 23, 26,3, 42968
    # 35754_011
    # 55703_004_footnotes
    # 30751_005 footnoretsa
    # 45823_008 footnotes
    # 55703_005_foothn
    # 48366_004
    # 47372_001
    # 54174_003
    # 327000_001
    # 8303_001
    # 54813_008 footntoes
    if not debug:
        for root, dirs, files in os.walk(pdf_path):
            for name in files:
                pdf_name = os.path.join(root, name)
                text_name = os.path.join(txt_folder, name.replace(".pdf", ".txt"))
                footnote_text_name = os.path.join(txt_folder, f"footnotes_{name}".replace(".pdf", ".txt"))
                main(pdf_name, text_name, footnote_text_name, output_folder)
    else:
        print("Debug mode")
        for pdf_name, single_file_txt, single_file_txt_fn in zip(debug_pdf_fnames, debug_txt_fnamse,
                                                                 debug_txt_fn_fnamse):
            main(pdf_name, single_file_txt, single_file_txt_fn, output_folder)
