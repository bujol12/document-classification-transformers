import argparse
import json
import matplotlib.pyplot as plt


def main():
    # Parse command line args
    args = parse_args()
    print("Preprocessing files...")

    # Open the file
    annotations_cnt = [0, 0, 0, 0, 0]
    labels = [[], [], [], [], []]
    with open(args.input_json) as data:
        for cnt, line in enumerate(data):
            line = json.loads(line)
            # print(line["x"], line["y"])

            # get annotations count
            annotations_cnt[0] += len(line["0"])
            annotations_cnt[1] += len(line["1"])
            annotations_cnt[2] += len(line["2"])
            annotations_cnt[3] += len(line["3"])
            annotations_cnt[4] += len(line["4"])

            # add labels
            labels[0].append(line["y"][0])
            labels[1].append(line["y"][1])
            labels[2].append(line["y"][2])
            labels[3].append(line["y"][3])
            labels[4].append(line["y"][4])

    print("Total annotations counts:", annotations_cnt)
    fig, ax = plt.subplots(2, 3, sharey=True)
    ax[0][0].hist(labels[0], bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], range=(0.0, 1.0))
    ax[0][0].set_title("Aspect 0")

    ax[0][1].hist(labels[1], bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], range=(0.0, 1.0))
    ax[0][1].set_title("Aspect 1")

    ax[0][2].hist(labels[2], bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], range=(0.0, 1.0))
    ax[0][2].set_title("Aspect 2")

    ax[1][0].hist(labels[3], bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], range=(0.0, 1.0))
    ax[1][0].set_title("Aspect 3")

    ax[1][1].hist(labels[4], bins=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], range=(0.0, 1.0))
    ax[1][1].set_title("Aspect 4")

    plt.tight_layout()
    plt.show()


def parse_args():
    # Parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input_json",
        help="Path to a JSON file, one JSON beer review per line")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
