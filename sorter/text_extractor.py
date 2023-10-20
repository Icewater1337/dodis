from pathlib import Path
from urllib.request import urlopen
from bs4 import BeautifulSoup

# Opening the html file

def parse_html_file(html_file):
    # Reading the file
    html_file_open = open(html_file, 'r',  encoding="utf-8")
    html = html_file_open.read()


    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

if __name__ == "__main__":
    base_path = "/home/fuchs/Desktop/dodis/dodo/docs/it/html"
    output_path = "/home/fuchs/Desktop/dodis/dodo/docs/it/text/"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    for html_file in Path(base_path).iterdir():
        if html_file.is_file():
            print("parse FIle: ", html_file)
            parsed_html = parse_html_file(html_file)
            with open(output_path + html_file.name.replace(html_file.suffix, ".txt"), "w") as file:
                file.write(parsed_html)