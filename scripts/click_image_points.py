import argparse
from pathlib import Path

import cv2


WINDOW_NAME = "image_click_points"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open an image and print the pixel coordinates when clicked."
    )
    parser.add_argument("image_path", type=Path, help="Path to the input image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = args.image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to open image: {image_path}")

    display_image = image.copy()

    def on_mouse(event: int, x: int, y: int, flags: int, param) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        print(f"Clicked pixel: ({x}, {y})", flush=True)
        cv2.circle(display_image, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(
            display_image,
            f"({x}, {y})",
            (x + 8, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_NAME, display_image)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"Opened image: {image_path}")
    print("Left click to print pixel coordinates. Press 'q' or Esc to exit.")

    while True:
        cv2.imshow(WINDOW_NAME, display_image)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
