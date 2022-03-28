import json
import argparse
from copy import deepcopy


def main():
    args = parse_args()

    output = {"documents": []}
    for file in args.i:
        print(file)
        with open(file) as fhand:
            inp = json.load(fhand)
            output["documents"] += deepcopy(inp["documents"])

    with open(args.out, 'w') as file:
        json.dump(output, file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple datasets into 1",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s -i <input-file> -i <input-file> ... -out <out_name>")

    parser.add_argument(
        "-i",
        help="Path to a JSON files",
        action='append')

    parser.add_argument(
        "-out",
        help="Path to output JSON file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
