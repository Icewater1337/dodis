import os
import shutil


def move_pdfs(src_dir, dest_dir):
    # Ensure the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through all files and directories in the source directory
    for foldername, subfolders, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                src_path = os.path.join(foldername, filename)
                dest_path = os.path.join(dest_dir, filename)

                # Ensure there's no file name collision in the destination directory
                counter = 1
                while os.path.exists(dest_path):
                    base, ext = os.path.splitext(filename)
                    dest_path = os.path.join(dest_dir, f"{base}_{counter}{ext}")
                    counter += 1

                # Move the PDF file to the destination directory
                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")


# Example usage:
src_folder = "/home/fuchs/Desktop/dodis/dodo/docs/it/sorted/"
dest_folder = "/home/fuchs/Desktop/dodis/dodo/docs/it/pdf/"
move_pdfs(src_folder, dest_folder)
