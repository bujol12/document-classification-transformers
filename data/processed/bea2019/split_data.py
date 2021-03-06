import json
import argparse
import random


def main():
    random.seed(40)
    args = parse_args()

    input_json = None
    with open(args.i) as f:
        input_json = json.load(f)

    if args.dev_idx_file is None:
        idx = list(range(0, len(input_json["documents"])))
        random.shuffle(idx)
        idx_count = int(len(idx) * 0.20)  # 20% goes to dev
        dev_idx = idx[:idx_count]
    else:
        with open(args.dev_idx_file) as f:
            dev_idx = json.load(f)

    train_docs = [input_json["documents"][idx] for idx in range(len(input_json["documents"])) if idx not in dev_idx]
    dev_docs = [input_json["documents"][idx] for idx in range(len(input_json["documents"])) if idx in dev_idx]


    # Calculate evidence distribution per class on the dev dataset
    pos_token_cnt = 0
    pos_evidence_cnt = 0
    neg_token_cnt = 0
    neg_evidence_cnt = 0

    token_cnt = 0
    evidence_cnt = 0

    for idx, doc in enumerate(dev_docs):
        if doc["document_label"] == 0:
            neg_token_cnt += sum([len(sent) for sent in doc["token_labels"]])
            neg_evidence_cnt += sum([sum(sent) for sent in doc["token_labels"]])
        else:
            pos_token_cnt += sum([len(sent) for sent in doc["token_labels"]])
            pos_evidence_cnt += sum([sum(sent) for sent in doc["token_labels"]])

        token_cnt += sum([len(sent) for sent in doc["token_labels"]])
        evidence_cnt += sum([sum(sent) for sent in doc["token_labels"]])

    print(f"Average Evidence Pct: {round(100 * evidence_cnt / token_cnt)}%")
    print(f"Average Evidence Pct: {round(100 * pos_evidence_cnt / pos_token_cnt)}% (Positive Class)")
    print(f"Average Evidence Pct: {round(100 * neg_evidence_cnt / neg_token_cnt)}% (Negative Class)")

    # store idx stored in the dataset
    dev_real_idx = [input_json["documents"][idx]["id"] for idx in range(len(input_json["documents"])) if
                    idx in dev_idx]

    train = {"documents": train_docs}
    dev = {"documents": dev_docs}

    with open(args.out_dev, 'w') as f:
        json.dump(dev, f)

    with open(args.out_train, 'w') as f:
        json.dump(train, f)

    with open(args.dev_idx_file_out, 'w') as f:
        json.dump(dev_real_idx, f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple datasets into 1",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s -i <input-file> -i <input-file> ... -out <out_name>")

    parser.add_argument(
        "-i",
        help="Path to a JSON file")

    parser.add_argument(
        "-out_train",
        help="Path to output train JSON file")

    parser.add_argument(
        "-out_dev",
        help="Path to output train JSON file")

    parser.add_argument(
        "-dev_idx_file",
        help="Path to file containing list of indices to choose for dev dataset"
    )

    parser.add_argument(
        "-dev_idx_file_out",
        help="Path to file containing list of indices choosen for dev dataset"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
