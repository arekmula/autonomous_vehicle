import pandas as pd
from argparse import ArgumentParser


def save_new_labels(df_labels: pd.DataFrame, filename="new_labels.csv"):
    df_labels.to_csv(filename, index_label="img_name")


def move_labels(df_labels: pd.DataFrame):

    df_moved_labels = pd.DataFrame(index=df_labels.index[:-1], columns=df_labels.columns)
    df_moved_labels[["steer", "steer_angle", "velocity"]] = df_labels[["steer", "steer_angle", "velocity"]].values[1:]
    save_new_labels(df_moved_labels, filename="first_dataset.csv")
    print(df_moved_labels)



def main(args):
    labels_path = args.labels_path

    df_labels = pd.read_csv(labels_path, index_col="img_name")
    move_labels(df_labels)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--labels_path", type=str, required=True)

    args, _ = parser.parse_known_args()
    main(args)