import os
import re
import shutil


def move_pdfs(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for foldername, subfolders, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.lower().endswith('.pdf') and not filename.lower().endswith("-dds.pdf"):
                adjusted_filename = re.findall(r'\d+', filename)
                adjusted_filename = "".join(adjusted_filename) + ".pdf"
                src_path = os.path.join(foldername, filename)
                dest_path = os.path.join(dest_dir, adjusted_filename)


                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")


# Example usage:
src_folder = "/media/fuchs/d/dodis_s3/pdf_all/"
dest_folder = "/media/fuchs/d/dodis_s3/pdf_no_dd/"
move_pdfs(src_folder, dest_folder)
