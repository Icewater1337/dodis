from pathlib import Path
from bs4 import BeautifulSoup

from utils.html_util import extract_text_from_soup


def parse_html_file(html_file):
    with open(html_file, 'r',  encoding="utf-8") as html_file_open:
        html = html_file_open.read()

    soup = BeautifulSoup(html, features="html.parser")

    # Extracting text from div with class 'footnotes'
    footnotes_div = soup.find('div', class_='footnotes')
    if footnotes_div:
        footnotes_text = extract_text_from_soup(footnotes_div)
        # Remove the footnotes div from the soup after extracting its text
        footnotes_div.extract()
    else:
        footnotes_text = None

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    main_text = extract_text_from_soup(soup)

    return main_text, footnotes_text


if __name__ == "__main__":
    base_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/html/"
    output_path = "/home/fuchs/Desktop/dodis/dodo/docs_p1/text_transcripts/"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for html_file in Path(base_path).iterdir():
        if html_file.is_file():
            print("parse FIle: ", html_file)
            main_text, footnotes_text = parse_html_file(html_file)

            with open(output_path + html_file.name.replace(html_file.suffix, ".txt"), "w") as file:
                file.write(main_text)

            if footnotes_text:
                with open(output_path + 'footnotes_' + html_file.name.replace(html_file.suffix, ".txt"), "w") as file:
                    file.write(footnotes_text)
