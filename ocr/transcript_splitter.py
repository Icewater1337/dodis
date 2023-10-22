import os

from fuzzywuzzy import fuzz
from pdf2image import convert_from_path
from pytesseract import image_to_string


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


def find_matching_index(ocrd, transcript):
    ocrd_words = ocrd.split()
    transcript_words = transcript.split()

    # Look for exact match
    for j in range(len(ocrd_words) - 2):
        for i in range(len(transcript_words) - 2):
            if transcript_words[i:i + 3] == ocrd_words[j:j + 3]:
                return j, i

    print("No exact match found. Attempting fuzzy match...")
    max_ratio = 0
    index = -1
    for j in range(len(ocrd_words) - 2):
        for i in range(len(transcript_words) - 2):
            ratio = fuzz.ratio(" ".join(transcript_words[i:i + 3]), " ".join(ocrd_words[j:j + 3]))
            if ratio > max_ratio:
                max_ratio = ratio
                index = i
    if max_ratio > 80:  # Threshold for approximate match. Adjust as needed.
        return j, index

    return -1, -1  # Return -1 if no match found

def convert_pdf_to_images(pdf_path):
    # Convert the PDF to a list of PIL images
    return convert_from_path(pdf_path, dpi=300)

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

import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_text_block(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Exclude the top 30% of the image
    exclude_top_percentage = 0.1
    start_y = int(exclude_top_percentage * gray.shape[0])



    # Remaining image after excluding top
    gray = gray[start_y:, :]
    plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
    plt.title("Image after Excluding Top Part")
    plt.show()

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
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        return

    boxes = [cv2.boundingRect(contour) for contour in valid_contours]
    merged_boxes_list = merge_boxes(boxes)  # Assuming you have this function

    main_text_block = choose_main_text_block(merged_boxes_list, img.shape[1],
                                             img.shape[0])  # Assuming you have this function
    x, y, w, h = main_text_block
    padding = 100
    x_min = max(0, x - padding)
    y_min = max(0, y - padding + start_y)  # Adjust the y-coordinate with the excluded portion
    x_max = min(img.shape[1], x + w + padding)
    y_max = min(img.shape[0], y + h + padding + start_y)  # Adjust the y-coordinate with the excluded portion
    cropped = img[y_min:y_max, x_min:x_max]

    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Text Block")
    plt.show()

    return cropped



def extract_text_from_image(image_path):
    return image_to_string(image_path, lang='deu')


def find_closest_match(target, source):
    highest_score = -1
    match_idx = -1
    for i in range(len(source) - len(target) + 1):
        current_score = fuzz.ratio(" ".join(target), " ".join(source[i:i+len(target)]))
        if current_score > highest_score:
            highest_score = current_score
            match_idx = i
    return match_idx

def extract_corresponding_transcript(ocr_text, full_transcript):
    ocr_words = ocr_text.split()

    transcript_words = full_transcript.split()

    ocr_idx, transcript_idx = find_matching_index(ocr_text, full_transcript)

    if ocr_idx == -1 or transcript_idx == -1:
        return None, full_transcript

    ocr_words = ocr_words[ocr_idx:]
    transcript_words = transcript_words[transcript_idx:]

    start_target = ocr_words[:3]
    start_match_idx = " ".join(transcript_words).find(" ".join(start_target))

    # If exact match doesn't exist, use fuzzy match
    if start_match_idx == -1:
        start_match_idx = find_closest_match(start_target, transcript_words)

    # Attempt exact match for the last three words
    end_target = ocr_words[-3:]
    end_match_idx = " ".join(transcript_words).find(" ".join(end_target))

    # If exact match doesn't exist, use fuzzy match
    if end_match_idx == -1:
        end_match_idx = find_closest_match(end_target, transcript_words)

    if start_match_idx == -1 or end_match_idx == -1:
        return None, full_transcript

    # Extract the corresponding text from the transcript
    matched_text = ' '.join(transcript_words)[start_match_idx:end_match_idx + len(" ".join(end_target))]
    remaining_transcript = ' '.join(transcript_words)[end_match_idx + len(" ".join(end_target)):]

    return matched_text, remaining_transcript

def main(pdf_path,text_path,output_folder):
    with open(text_path, 'r') as f:
        full_transcript = f.read()

    # images = convert_pdf_to_images(pdf_path)
    images = convert_from_path(pdf_path, dpi=300)
    # images = [auto_crop(img) for img in images]

    txt_filename = os.path.basename(text_path)
    pdf_filename = os.path.basename(pdf_path)
    for idx, image in enumerate(images, start=1):
        image = find_text_block(image)
        ocr_text = extract_text_from_image(image)

        matched_text, full_transcript = extract_corresponding_transcript(ocr_text, full_transcript)

        if matched_text is None:
            print(f"Couldn't match text for pdf  {pdf_path} page {idx}.")
            continue

        with open(f"{txt_filename}_{idx:03}.txt", 'w') as f:
            f.write(matched_text)

        # Optionally save the image
        image.save(f"{pdf_filename}_{idx:03}.png", "PNG")


if __name__ == "__main__":
    pdf_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/de/year_sorted/computer/"
    txt_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/de/txt/"
    output_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/de/split_output/"
    for root, dirs, files in os.walk(pdf_path):
        for name in files:
            pdf_name = os.path.join(root, name)
            text_name = os.path.join(txt_folder, name.replace(".pdf", ".txt"))
            main(pdf_name,text_name,output_folder)
