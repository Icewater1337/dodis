import os


def count_files_in_folder(root_dir):
    total_copunt = 0
    for foldername, subfolders, filenames in os.walk(root_dir):
        num_files = len(filenames)
        total_copunt += num_files
        print(f"Folder: {foldername}, Number of Files: {num_files}")
    print(f"Total number of files: {total_copunt}")



root_directory = "/home/fuchs/Desktop/dodis/dodo/docs_p1/sorted/en/year_sorted/computer/"
count_files_in_folder(root_directory)