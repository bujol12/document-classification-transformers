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
    empty_doc = {"tokens": [], "document_label": None, "sentence_labels": None, "token_labels": [], "id": None}

    # process files
    with open(args.csv_file) as f:
        cnt = 0
        for line in f:
            label, text = line.split("\t")
            # remove annotations
            # move token annotations and last full stop to avoid splitting into an empty one
            text = text.replace(" </POS>", "").replace("<POS> ", "").replace(" </NEG>", "").replace("<NEG> ",
                                                                                                    "").strip()
            sentences = text.split(" . ")  # split into sentences. Doesn't split final . due to spaces
            # remove trailing spaces and newlines + add end of sentence dots back
            sentences = [sent.strip().split(" ") for sent in sentences]

            doc = deepcopy(empty_doc)
            doc["tokens"] = sentences
            doc["id"] = cnt
            cnt += 1
            doc["document_label"] = int(label)
            output_dict["documents"].append(doc)

    with open(args.block_csv_file) as f:
        i = 0
        for line in f:
            j = 0
            token_labels_doc = [int(label) for label in line.strip().split(" ")]

            output_dict["documents"][i]["token_labels"] = []
            for sent in output_dict["documents"][i]["tokens"]:
                token_labels = token_labels_doc[j:j + len(sent)]
                j += len(sent)
                output_dict["documents"][i]["token_labels"].append(token_labels)

            i += 1

    pos_output_dict = deepcopy(output_dict)
    neg_output_dict = deepcopy(output_dict)

    sum_evidence_tokens = 0
    total_tokens = 0
    for doc in pos_output_dict["documents"]:
        if doc["document_label"] == 0:  # zero-out negative ones
            doc["token_labels"] = [[0 for _ in sent] for sent in doc["token_labels"]]

        # test if labels for tokens are of the same size and tokens themselves
        for sent_id in range(len(doc["token_labels"])):
            if doc["document_label"] == 1:
                sum_evidence_tokens += sum(doc["token_labels"][sent_id])
                total_tokens += len(doc["token_labels"][sent_id])

            if len(doc["token_labels"][sent_id]) != len(doc["tokens"][sent_id]):
                print("non-matching sizes!")
                print("tokens_size:", len(doc["tokens"][sent_id]))
                print("labels_size:", len(doc["token_labels"][sent_id]))
                print(doc["tokens"][sent_id])
                print(sent_id)
                return

    print(f"Average Evidence Pct: {round(100* sum_evidence_tokens / total_tokens)}% (Pos Dataset)")

    sum_evidence_tokens = 0
    total_tokens = 0
    for doc in neg_output_dict["documents"]:
        if doc["document_label"] == 1:  # zero-out positive ones
            doc["document_label"] = 0
            doc["token_labels"] = [[0 for _ in sent] for sent in doc["token_labels"]]
        else:  # swap labels around so that negative becomes positive
            doc["document_label"] = 1
            for sent in doc["token_labels"]:
                sum_evidence_tokens += sum(sent)
                total_tokens += len(sent)


    print(f"Average Evidence Pct: {round(100* sum_evidence_tokens / total_tokens)}% (Neg Dataset)")

    with open(args.pos_out, 'w') as f:
        json.dump(pos_output_dict, f)

    with open(args.neg_out, 'w') as f:
        json.dump(neg_output_dict, f)


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
