import sys
import json

if __name__ == "__main__":
    file = sys.argv[1]

    with open(file) as fhand:
        data = json.load(fhand)

    for doc_id, doc in enumerate(data):
        for i in range(len(doc["tokens"])):
            if not isinstance(doc["tokens"][i], list):
                if doc["tokens"][i] not in ("<pad>", "</s>"):
                    token, pred, label = doc["tokens"][i], doc["token_preds"][i], doc["label_ids"][i]
                    print(f"{token}\t{pred}\t{label}")
            else:
                for j in range(len(doc["tokens"][i])):
                    if doc["tokens"][i][j] not in ("<pad>", "</s>"):
                        token, pred, label = doc["tokens"][i][j], doc["token_preds"][i][j], doc["label_ids"][i][j]
                        print(f"{token}\t{pred}\t{label}")
                print()
        print()