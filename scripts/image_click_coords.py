import argparse
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "image_click_coords"
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 960


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an image and print clicked pixel coordinates."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    return parser.parse_args()


def load_image(image_path: Path) -> np.ndarray:
    file_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to open image: {image_path}")
    return image


def resize_to_fit(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(MAX_DISPLAY_WIDTH / width, MAX_DISPLAY_HEIGHT / height, 1.0)
    if scale == 1.0:
        return image.copy(), scale

    resized = cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def main() -> None:
    image_path = r"F:\763\反 (4 5 6 7)\images\9-4 (11).PNG"

    canvas = load_image(image_path)
    display_canvas, scale = resize_to_fit(canvas)

    def handle_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        original_x = min(max(int(round(x / scale)), 0), canvas.shape[1] - 1)
        original_y = min(max(int(round(y / scale)), 0), canvas.shape[0] - 1)

        print(f"Clicked pixel: ({original_x}, {original_y})", flush=True)
        cv2.circle(canvas, (original_x, original_y), 4, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"({original_x}, {original_y})",
            (original_x + 8, max(original_y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        resized_canvas, _ = resize_to_fit(canvas)
        # cv2.imshow(WINDOW_NAME, resized_canvas)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse)

    print(f"Opened image: {image_path}")
    print("Left click to print coordinates. Press 'q' or Esc to exit.")

    while True:
        cv2.imshow(WINDOW_NAME, display_canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
