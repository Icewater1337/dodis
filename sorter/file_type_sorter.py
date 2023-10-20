from distutils.file_util import move_file
from pathlib import Path


def create_folders(base_path):
    json_path = base_path + "json"
    html_path = base_path + "html"
    pdf_path = base_path + "pdf"
    folders = [json_path,html_path,pdf_path]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def sort_by_file_type(base_path):
    for file in Path(base_path).iterdir():
        if file.is_file():
            file_type = file.suffix
            Path(str(file)).rename(base_path + file_type[1:] + "/" + file.name)
    return


if __name__ == "__main__":
    base_path = "/home/fuchs/Desktop/dodis/dodo/docs/it/"
    # base_path = "/home/fuchs/Desktop/dodis/dodo/docs/output/de/"
    create_folders(base_path)
    sort_by_file_type(base_path)
