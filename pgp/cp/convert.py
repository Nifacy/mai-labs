import argparse
import pathlib
import numpy as np

from PIL import Image


def read_data_file(input_path: pathlib.Path) -> Image.Image:
    with input_path.open("rb") as file:
        width = int.from_bytes(file.read(4), byteorder="little")
        height = int.from_bytes(file.read(4), byteorder="little")

        raw_data = file.read()
        img_data = np.frombuffer(raw_data, dtype=np.uint8)
        img_data = img_data.reshape((height, width, 4))

    return Image.fromarray(img_data[:, :, :3], "RGB")


def save_as_jpg(image: Image.Image, output_path: pathlib.Path) -> None:
    image.save(str(output_path.absolute()), "JPEG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .data image file to JPG format.",
    )

    parser.add_argument("input_files_mask", help="Path or mask to the input .data file.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_files_mask = args.input_files_mask

    for input_file in pathlib.Path(".").rglob(input_files_mask):
        output_file = pathlib.Path(f"{input_file}.jpg")

        print(f"[log] process {input_file} ...")
        save_as_jpg(read_data_file(input_file), output_file)
        print(f"[log] image {output_file} created")
