import os
import shutil
from pathlib import Path


def move_pdfs(src_dir, dest_dir):
    # Ensure the destination directory exists
    dest_dir_png = dest_dir + "png/"
    dest_dir_txt = dest_dir + "txt/"
    Path(dest_dir_png).mkdir(parents=True, exist_ok=True)
    Path(dest_dir_txt).mkdir(parents=True, exist_ok=True)

    # Walk through all files and directories in the source directory
    for foldername, subfolders, filenames in os.walk(src_dir):
        lang = foldername.split('/')[-1]
        for filename in filenames:
            if filename.lower().endswith('.png'):
                dest_png_name = filename.replace('.png', f'_{lang}.png')
                txt_file = filename.replace('.png', '.txt')
                dest_txt_name = txt_file.replace('.txt', f'_{lang}.txt')


                src_path_png = os.path.join(foldername, filename)
                src_path_txt = os.path.join(foldername, txt_file)

                dest_path_png = os.path.join(dest_dir_png, dest_png_name)
                dest_path_txt = os.path.join(dest_dir_txt, dest_txt_name)

                # Ensure there's no file name collision in the destination directory
                counter = 1
                while os.path.exists(dest_path_png):
                    base, ext = os.path.splitext(filename)
                    dest_path_png = os.path.join(dest_dir_png, f"{base}_{counter}{ext}")
                    counter += 1
                while os.path.exists(dest_path_txt):
                    base, ext = os.path.splitext(filename)
                    dest_path_txt = os.path.join(dest_dir_txt, f"{base}_{counter}{ext}")
                    counter += 1

                # Move the PDF file to the destination directory
                shutil.copy(src_path_png, dest_path_png)
                shutil.copy(src_path_txt, dest_path_txt)
                print(f"Copied: {src_path_png} -> {dest_path_png}")
                print(f"Copied: {src_path_txt} -> {dest_path_txt}")


# Example usage:
src_folder = "/media/fuchs/d/dataset_try_4/parts/"
dest_folder = "/media/fuchs/d/dataset_try_4/final_dataset/"
move_pdfs(src_folder, dest_folder)
