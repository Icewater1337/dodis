import os
from bs4 import BeautifulSoup


def check_divs_in_html(file_path):
    footnote = False
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        soup = BeautifulSoup(content, 'html.parser')

        tei_div = soup.find('div', class_='tei-div tei-div')
        if not tei_div:
            print(f'For document {file_path} the "tei-div tei-div" div does not exist.')
            return None
        # Check for div with id 'footnotes'
        footnotes_div = soup.find('div', class_='footnotes')
        if footnotes_div:
            footnote = True
        else:
            print(f'For document {file_path} the "tei-div tei-div" div does not exist.')

        # Check for div with class 'tei-div tei-div'
        tei_div = soup.find('div', class_='tei-div tei-div')
        if not tei_div:
            print(f'For document {file_path} the "tei-div tei-div" div does not exist.')
    return footnote

def main():
    folder_path = input("Enter the folder path containing the HTML documents: ")
    footnote_counter = 0
    no_footnote_counter = 0

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.html'):
            footnote = check_divs_in_html(os.path.join(folder_path, file_name))
            if footnote is None:
                os.remove(os.path.join(folder_path, file_name))
                print(f'Removed {file_name}')
                continue
            if footnote:
                footnote_counter += 1
            else:
                no_footnote_counter += 1
    print(f'Number of documents with footnotes: {footnote_counter}')
    print(f'Number of documents without footnotes: {no_footnote_counter}')


if __name__ == "__main__":

    main()
