import os

from fuzzywuzzy import fuzz
from pdf2image import convert_from_path
from pytesseract import image_to_string


import cv2
import numpy as np
from matplotlib import pyplot as plt


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

def find_text_block(img):
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 500
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    if not valid_contours:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()
        return

    boxes = [cv2.boundingRect(contour) for contour in valid_contours]
    merged_boxes_list = merge_boxes(boxes)

    main_text_block = choose_main_text_block(merged_boxes_list, img.shape[1], img.shape[0])
    x, y, w, h = main_text_block
    padding = 50
    x_min = max(0,x-padding)
    y_min = max(0,y-padding)
    x_max = min(img.shape[1],x+w+padding)
    y_max = min(img.shape[0],y+h+padding)
    # cropped = img[y:y + h, x:x + w]
    cropped = img[y_min:y_max, x_min:x_max]
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.show()
    return cropped

def convert_pdf_to_images(pdf_path):
    # Convert the PDF to a list of PIL images
    return convert_from_path(pdf_path)

def extract_text_from_image(image_path):
    return image_to_string(image_path)


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

    # Attempt exact match for the first three words
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

    images = convert_pdf_to_images(pdf_path)
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
