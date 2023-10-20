import os

def rename_files(folder_path):
    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file has a '.pdf' extension and contains '_1'
        if filename.endswith('_1.pdf'):
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, filename.replace('_1', ''))
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {filename.replace("_1", "")}')

if __name__ == "__main__":
    path_to_folder = "/home/fuchs/Desktop/dodis/dodo/docs_p1/pdf/"  # Provide the path to your folder containing the pdf files
    rename_files(path_to_folder)
