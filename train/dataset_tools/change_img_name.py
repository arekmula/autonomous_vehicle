import pandas as pd
from argparse import ArgumentParser


def save_new_labels(df_labels: pd.DataFrame, filename="new_labels.csv"):
    df_labels.to_csv(filename, index_label="img_name")


def append_to_img_name(df_labels: pd.DataFrame, appendix: str = "2_"):

    img_names = df_labels.index

    appendix_column = [appendix] * len(img_names)
    appendix_series = pd.Series(appendix_column)

    df_labels.index = appendix_series + img_names

    save_new_labels(df_labels)


def main(args):
    labels_path = args.labels_path

    df_labels = pd.read_csv(labels_path, index_col="img_name")
    append_to_img_name(df_labels)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--labels_path", type=str, required=True)

    args, _ = parser.parse_known_args()
    main(args)