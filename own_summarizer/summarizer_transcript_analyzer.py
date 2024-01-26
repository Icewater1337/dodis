

import os

from utils.folder_util import load_file_contents_in_folder


def sort_by_value_length(files_dict):
    """
    Sort a dictionary by the length of the values.
    :param files_dict: Dictionary to sort
    :return: Sorted dictionary
    """
    return {k: v for k, v in sorted(files_dict.items(), key=lambda item: len(item[1]))}


def print_n_shortest_files(sorted_dict, param):
    """
    Print the n shortest files from a sorted dictionary.
    :param sorted_dict: Sorted dictionary
    :param param: Number of files to print
    """
    print(f"Printing {param} shortest files:")
    for i, (name, content) in enumerate(sorted_dict.items()):
        if i >= param:
            break
        print(f"{name}: with length {len(content)}")


def print_n_middle_files(sorted_dict, param):
    """
    Print the n middle files from a sorted dictionary.
    :param sorted_dict: Sorted dictionary
    :param param: Number of files to print
    """
    print(f"Printing {param} middle files:")
    it = (len(sorted_dict) // 2) - (param // 2)
    it_end = it + param
    while it < it_end:
        name = list(sorted_dict.keys())[it]
        content = list(sorted_dict.values())[it]
        print(f"{name}: with length {len(content)}")
        it += 1

def print_n_longest_files(sorted_dict, param):
    """
    Print the n longest files from a sorted dictionary.
    :param sorted_dict: Sorted dictionary
    :param param: Number of files to print
    """
    print(f"Printing {param} longest files:")
    for i, (name, content) in enumerate(sorted_dict.items()):
        if i < len(sorted_dict) - param:
            continue
        print(f"{name}: with length {len(content)}")


def print_length_quartiles(sorted_dict):
    """
    Print the quartiles of the file lengths from a sorted dictionary.
    :param sorted_dict: Sorted dictionary
    """
    print("Printing length quartiles:")
    print(f"Q1: {len(list(sorted_dict.values())[len(sorted_dict) // 4])}")
    print(f"Q2: {len(list(sorted_dict.values())[len(sorted_dict) // 2])}")
    print(f"Q3: {len(list(sorted_dict.values())[len(sorted_dict) // 4 * 3])}")


if __name__=="__main__":
    # files_dict = load_file_contents_in_folder("/home/fuchs/Desktop/dodis/dodo/docs_p1/summarizer_transcripts", file_type="txt", return_dict=True)
    files_dict = load_file_contents_in_folder("../inputs_bak", file_type="txt", return_dict=True)
    sorted_dict = sort_by_value_length(files_dict)

    print_length_quartiles(sorted_dict)
    print_n_shortest_files(sorted_dict, 10)
    # print_n_middle_files(sorted_dict, 10)
    # print_n_longest_files(sorted_dict, 10)
