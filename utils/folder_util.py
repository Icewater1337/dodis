from pathlib import Path


def load_file_contents_in_folder(folder_path, file_type = None, return_dict = False):
    """
    Load all files in a folder and return a list of their contents.
    :param folder_path: Path to the folder
    :param file_type: File type to load
    :param return_dict: If True, return a dictionary with the file name as key and the file content as value
    :return: List of file contents
    """
    file_contents = []
    file_name_dict = {}
    for file in Path(folder_path).iterdir():
        if file_type is None or file.suffix == f".{file_type}":
            file_contents.append(file.read_text(encoding="utf-8"))
            if return_dict:
                file_name_dict[file.name] = file.read_text(encoding="utf-8")
    if return_dict:
        return file_name_dict
    return file_contents

