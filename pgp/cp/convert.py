import argparse
import numpy as np

from dataclasses import dataclass
from PIL import Image


@dataclass
class _Image:
    width: int
    height: int
    data: np.ndarray[int]


def read_data_file(filepath: str) -> _Image:
    with open(filepath, "rb") as file:
        width = int.from_bytes(file.read(4), byteorder="little")
        height = int.from_bytes(file.read(4), byteorder="little")

        raw_data = file.read()
        img_data = np.frombuffer(raw_data, dtype=np.uint8)
        img_data = img_data.reshape((height, width, 4))

    return _Image(
        width=width,
        height=height,
        data=img_data,
    )


def save_as_jpg(image: _Image, output_path: str) -> None:
    rgb_image = Image.fromarray(image.data[:, :, :3], "RGB")
    rgb_image.save(output_path, "JPEG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .data image file to JPG format.",
    )

    parser.add_argument("input_file", help="Path to the input .data file.")
    parser.add_argument("output_file", nargs="?", help="Path to the output JPG file. Defaults to input_file with .jpg extension.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file if args.output_file else f"{input_file}.jpg"

    save_as_jpg(read_data_file(input_file), output_file)
