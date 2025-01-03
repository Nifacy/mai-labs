from dataclasses import dataclass
from typing import Iterable, Sequence
import convert
import pathlib
import argparse

import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image


@dataclass
class _RenderFrame:
    image: Image.Image
    k: int


def load_images(paths: Iterable[pathlib.Path]) -> list[_RenderFrame]:
    return [
        _RenderFrame(
            image=convert.read_data_file(image_path),
            k=int(image_path.stem),
        )
        for image_path in paths 
    ]    


def create_app(images: Sequence[_RenderFrame]) -> tk.Tk:
    root = tk.Tk()
    root.title("Viewer")

    prepaired_frames = [ 
        (el.k, ImageTk.PhotoImage(el.image))
        for el in images
    ]

    current_image = tk.Label(root)
    current_image.pack(pady=10)

    current_k_value = tk.Label(root)
    current_k_value.pack(pady=10)

    def update_image(value: str):
        index = int(float(value))
        curr_k, curr_image = prepaired_frames[index]

        current_k_value.config(text=f"k: {curr_k}")
        current_image.config(image=curr_image)
        current_image.image = curr_image

    # Slider
    slider = ttk.Scale(
        root,
        from_=0, to=len(images) - 1,
        orient="horizontal",
        command=update_image
    )
    slider.pack(pady=20, padx=20, fill="x")

    update_image('0')

    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a .data image file to JPG format.",
    )

    parser.add_argument("input_files_mask", help="Path or mask to the input .data file.")

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    input_files = pathlib.Path(".").rglob(args.input_files_mask)

    images = load_images(input_files)
    app = create_app(images)

    app.mainloop()
