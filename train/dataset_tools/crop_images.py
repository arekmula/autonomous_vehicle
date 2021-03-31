import cv2
from argparse import ArgumentParser
from pathlib import Path

Y_OFFSET = 288
Y_HEIGHT = 264


def crop_image(img_path: Path):
    img = cv2.imread(str(img_path))
    img_cropped = img[Y_OFFSET:Y_OFFSET+Y_HEIGHT, :]

    return img_cropped


def save_cropped_image(img_cropped, img_name: str, output_path: str):
    cv2.imwrite(output_path + "/" + img_name + ".png", img_cropped)
    print(img_name + ".png")


def main(args):
    dataset_path = Path(args.dataset_path)
    output_path = args.output_path

    images_paths = list(dataset_path.rglob("*.png"))
    for image_path in images_paths:
        img_cropped = crop_image(image_path)
        save_cropped_image(img_cropped, image_path.stem, output_path)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--dataset_path', type=str, help='Path to images', required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args, _ = parser.parse_known_args()
    main(args)
