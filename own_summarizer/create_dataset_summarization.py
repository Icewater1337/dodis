from pathlib import Path

from bs4 import BeautifulSoup

from utils.folder_util import load_file_contents_in_folder
from utils.html_util import extract_text_from_soup


def replace_footnote_references(main_div, footnotes_div):
    """
    Replace footnote references in the main text with the actual footnote content.
    """
    # Clone the main text div to avoid modifying the original
    main_div_clone = BeautifulSoup(str(main_div), 'html.parser')

    # Extract all footnote references from the main text
    footnote_refs = main_div_clone.find_all("a", class_="note")

    for ref in footnote_refs:
        # Extract the href attribute to find the corresponding footnote
        href = ref.get('href')
        if href:
            # Find the corresponding footnote by its ID
            footnote = footnotes_div.find('dl', id=href.strip("#"))
            if footnote:
                # Extract the footnote content
                footnote_content = footnote.find('dd', class_='fn-content')
                if footnote_content:
                    # Replace the reference in the main text with the footnote content
                    ref.replace_with(f"({footnote_content.get_text(strip=True)})")

    # Return the modified main text with integrated footnotes
    return main_div_clone.get_text(strip=True)

def extract_htmls(html_content):
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Locate the main text and the footnotes
    main_text_div = soup.find("div", class_="tei-div tei-div")
    footnotes_div = soup.find("div", class_="footnotes")


    return main_text_div, footnotes_div


def save_modified_text(name, modified_text, save_folder):
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    with open(save_folder + name.replace(".html", ".txt"), "w",  encoding='utf-8') as file:
        file.write(modified_text)


if __name__ == "__main__":
    output_folder = "C:\\Users\\Mathias\\Desktop\\dodis_project\\transcripts\\summarizer_transcripts\\"
    html_folder = "C:\\Users\\Mathias\\Desktop\\dodis_project\\transcripts\\html\\"
    html_files = load_file_contents_in_folder(html_folder, file_type="html", return_dict=True)
    for name, file_content  in html_files.items():
        main_text_div, footnotes_div = extract_htmls(file_content)
        if footnotes_div is None:
            modified_text = extract_text_from_soup(main_text_div)
        else:
            # Apply the function to replace footnote references
            modified_text = replace_footnote_references(main_text_div, footnotes_div)
        save_modified_text(name, modified_text, output_folder)
