import io
import subprocess
import os

import os
import subprocess
from PIL import Image
import PyPDF2


def fine_tune_modedl(output_directory, tessdata_directory):
    subprocess.run(["combine_tessdata", "-e", os.path.join(tessdata_directory, "eng.traineddata"), "eng.lstm"])
    subprocess.run(["lstmtraining",
                    "--model_output", os.path.join(output_directory, "result"),
                    "--continue_from", "eng.lstm",
                    "--traineddata", os.path.join(tessdata_directory, "eng.traineddata"),
                    "--train_listfile", "all-lstmf",
                    "--max_iterations", "4000"])

    subprocess.run(["lstmtraining",
                    "--stop_training",
                    "--continue_from", os.path.join(output_directory, "result_checkpoint"),
                    "--traineddata", os.path.join(tessdata_directory, "eng.traineddata"),
                    "--model_output", os.path.join(output_directory, "eng_final.traineddata")])




def convert_pdf_to_images(pdf_path, output_dir):
    # Open the PDF file
    with open(pdf_path, "rb") as pdf_file:
        pdf = PyPDF2.PdfReader(pdf_file)

        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]


            # Extracting images from the PDF
            xObject = page['/Resources']['/XObject'].get_object()
            for obj in xObject:
                if xObject[obj]['/Subtype'] == '/Image':
                    size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                    data = xObject[obj].get_data()
                    image = Image.open(io.BytesIO(data))

                    # Binarize the image
                    image = image.convert("L")
                    image = image.point(lambda x: 0 if x < 128 else 255, '1')

                    # Save the image in TIFF format
                    image_basename = os.path.basename(pdf_path).replace(".pdf", f"_page{page_num}.tif")
                    image.save(os.path.join(output_dir, image_basename))




# ... Rest of the Tesseract training steps


if __name__ == "__main__":
    output_directory = "/home/fuchs/Desktop/dodis/dodo/docs_p1/output"
    tessdata_directory = "/path_to_tesseract/tessdata/"
    pdf_directory = "/home/fuchs/Desktop/dodis/dodo/docs_p1/pdf/"

    for file in os.listdir(pdf_directory):
        if file.endswith(".pdf"):
            convert_pdf_to_images(os.path.join(pdf_directory, file), tessdata_directory)

    #Todo Problem: Images are multipaged, for training we need one page an transcripts.
    # What to o with these= do single image.

    # image_file = "path_to_image/image.tif"
    # subprocess.run(
    #     ["tesseract", image_file, image_file.rsplit('.', 1)[0], "-c", "tessedit_create_lstmbox=1", "lstm.train"])
    #
    # lstm_directory = "path_to_directory/"
    # with open("all-lstmf", "w") as f:
    #     for file in os.listdir(lstm_directory):
    #         if file.endswith(".lstmf"):
    #             f.write(os.path.join(lstm_directory, file) + "\n")
    #
    # fine_tune_modedl(output_directory, tessdata_directory)
    # output_text_file = "output.txt"
    # subprocess.run(["tesseract", image_file, output_text_file, "-l", "eng", "--tessdata-dir", output_directory])
