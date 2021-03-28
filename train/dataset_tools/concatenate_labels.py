import pandas as pd
from argparse import ArgumentParser


def save_new_labels(df_labels: pd.DataFrame, filename="labels_concatenated.csv"):
    df_labels.to_csv(filename, index_label="img_name")


def main(args):
    first_labels_path = args.first_labels_path
    second_labels_path = args.second_labels_path

    df_first_labels = pd.read_csv(first_labels_path, index_col="img_name")
    df_second_labels = pd.read_csv(second_labels_path, index_col="img_name")

    df_concatenated_labels = pd.concat([df_first_labels, df_second_labels])
    save_new_labels(df_concatenated_labels)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--first_labels_path", type=str, required=True)
    parser.add_argument("--second_labels_path", type=str, required=True)

    args, _ = parser.parse_known_args()
    main(args)