import argparse
import json
from copy import deepcopy
import pandas as pd


def main():
    """
    Convert Imdb reviews with labels 1 and 0 to Imdb-pos and Imdb-neg to support evidence extraction
    Remove evidence annotations from text and rely only on the block matrix provided
    """
    args = parse_args()

    output_dict = {"documents": []}
    empty_doc = {"tokens": [], "document_label": None, "sentence_labels": None, "token_labels": []}

    # process files
    with open(args.csv_file) as f:
        for line in f:
            label, text = line.split("\t")
            # remove annotations
            text = text.replace("</POS>", "").replace("<POS>", "").replace("</NEG>", "").replace("<NEG>", "")
            sentences = text.split(".") # split into sentences
            # remove trailing spaces and newlines + add end of sentence dots back
            sentences = [sent.strip().split(" ") + ["."] for sent in sentences]

            doc = deepcopy(empty_doc)
            doc["tokens"] = sentences
            doc["document_label"] = int(label)
            output_dict["documents"].append(doc)

    with open(args.block_csv_file) as f:
        i = 0
        for line in f:
            labels = [int(label) for label in line.strip().split(" ")]
            output_dict["documents"][i]["token_labels"] = labels
            i += 1
    # TODO splitting into positive and negative
    with open(args.pos_out, 'w') as f:
        json.dump(output_dict, f)


# Parse command line args
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert Imdb CSV file to internal JSON format.",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [-h] (-auto | -gold) [options] json_input_file -out <out_name>")
    parser.add_argument(
        "-csv_file",
        help="Path to a csv file, one CSV review per line.")
    parser.add_argument(
        "-block_csv_file",
        help="Path to a csv file, one CSV review per line.")
    parser.add_argument(
        "-pos_out",
        help="Path to positive output classification dataset")
    parser.add_argument(
        "-neg_out",
        help="Path to negative output classification dataset")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
