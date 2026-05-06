from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image


Connectivity = Literal[4, 8]
SortMode = Literal["row_major", "none"]


@dataclass(frozen=True)
class MaskToXYResult:
    """Result returned by mask_to_xy_points when metadata is requested."""

    points_xy: np.ndarray
    denoised_mask: np.ndarray
    original_foreground_pixels: int
    kept_foreground_pixels: int
    removed_components: int
    kept_components: int


def load_binary_mask(
    mask: str | Path | Image.Image | np.ndarray,
    *,
    threshold: int | float = 127,
    foreground_value: Literal["bright", "dark"] = "bright",
) -> np.ndarray:
    """Load a binary mask as a bool array.

    Args:
        mask: Image path, PIL image, or numpy array. RGB/RGBA images are converted
            to grayscale before thresholding.
        threshold: Pixel values greater than this are foreground when
            foreground_value is "bright".
        foreground_value: Use "bright" for normal masks where white is foreground,
            or "dark" for inverted masks where black is foreground.

    Returns:
        A bool array with shape [height, width].
    """

    if isinstance(mask, (str, Path)):
        array = np.asarray(Image.open(mask).convert("L"))
    elif isinstance(mask, Image.Image):
        array = np.asarray(mask.convert("L"))
    else:
        array = np.asarray(mask)

    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[:, :, :3]
        array = array.astype(np.float32).mean(axis=2)
    elif array.ndim != 2:
        raise ValueError(f"Mask must be 2D or image-like 3D array, got shape {array.shape}")

    if array.dtype == np.bool_:
        binary = array
    elif foreground_value == "bright":
        binary = array > threshold
    elif foreground_value == "dark":
        binary = array <= threshold
    else:
        raise ValueError("foreground_value must be 'bright' or 'dark'")

    return binary.astype(bool, copy=False)


def denoise_binary_mask(
    mask: str | Path | Image.Image | np.ndarray,
    *,
    threshold: int | float = 127,
    min_area: int = 16,
    connectivity: Connectivity = 8,
    keep_largest: bool = False,
    foreground_value: Literal["bright", "dark"] = "bright",
    return_metadata: bool = False,
) -> np.ndarray | MaskToXYResult:
    """Remove small connected foreground components from a binary mask.

    The denoising rule is area-based: each connected white region is measured in
    pixels, and regions smaller than min_area are removed. This removes isolated
    speckles without changing larger target regions.
    """

    if min_area < 1:
        raise ValueError("min_area must be >= 1")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    binary = load_binary_mask(
        mask,
        threshold=threshold,
        foreground_value=foreground_value,
    )
    denoised, removed_components, kept_components = _remove_small_components(
        binary,
        min_area=min_area,
        connectivity=connectivity,
        keep_largest=keep_largest,
    )

    if not return_metadata:
        return denoised

    points = _mask_to_points(denoised, sort="row_major")
    return MaskToXYResult(
        points_xy=points,
        denoised_mask=denoised,
        original_foreground_pixels=int(binary.sum()),
        kept_foreground_pixels=int(denoised.sum()),
        removed_components=removed_components,
        kept_components=kept_components,
    )


def mask_to_xy_points(
    mask: str | Path | Image.Image | np.ndarray,
    *,
    threshold: int | float = 127,
    min_area: int = 16,
    connectivity: Connectivity = 8,
    keep_largest: bool = False,
    foreground_value: Literal["bright", "dark"] = "bright",
    sort: SortMode = "row_major",
    return_metadata: bool = False,
) -> np.ndarray | MaskToXYResult:
    """Convert a binary mask to foreground pixel coordinates.

    Args:
        mask: Image path, PIL image, or numpy array.
        threshold: Foreground threshold for non-bool masks.
        min_area: Remove connected components smaller than this many pixels.
        connectivity: 4-neighbor or 8-neighbor connected component definition.
        keep_largest: If true, keep only the largest remaining component.
        foreground_value: "bright" means white foreground; "dark" means black foreground.
        sort: "row_major" returns points ordered by y then x. "none" returns numpy's
            natural np.where order, which is currently also row-major for 2D arrays.
        return_metadata: If true, return MaskToXYResult instead of only points.

    Returns:
        np.ndarray of shape [N, 2], dtype int32. Each row is [x, y].
    """

    result = denoise_binary_mask(
        mask,
        threshold=threshold,
        min_area=min_area,
        connectivity=connectivity,
        keep_largest=keep_largest,
        foreground_value=foreground_value,
        return_metadata=True,
    )
    assert isinstance(result, MaskToXYResult)
    points = _mask_to_points(result.denoised_mask, sort=sort)

    if return_metadata:
        return MaskToXYResult(
            points_xy=points,
            denoised_mask=result.denoised_mask,
            original_foreground_pixels=result.original_foreground_pixels,
            kept_foreground_pixels=result.kept_foreground_pixels,
            removed_components=result.removed_components,
            kept_components=result.kept_components,
        )

    return points


def mask_file_to_xy_points(
    mask_path: str | Path,
    **kwargs,
) -> np.ndarray | MaskToXYResult:
    """Read a mask image from disk and return foreground [x, y] coordinates."""

    return mask_to_xy_points(mask_path, **kwargs)


def mask_to_xy_list(
    mask: str | Path | Image.Image | np.ndarray,
    **kwargs,
) -> list[tuple[int, int]]:
    """Return foreground coordinates as a Python list of (x, y) tuples."""

    points = mask_to_xy_points(mask, **kwargs)
    if isinstance(points, MaskToXYResult):
        points = points.points_xy
    return [(int(x), int(y)) for x, y in points]


def _mask_to_points(mask: np.ndarray, *, sort: SortMode) -> np.ndarray:
    if sort not in ("row_major", "none"):
        raise ValueError("sort must be 'row_major' or 'none'")

    ys, xs = np.where(mask)
    points = np.stack((xs, ys), axis=1).astype(np.int32, copy=False)
    if sort == "row_major" and points.size:
        order = np.lexsort((points[:, 0], points[:, 1]))
        points = points[order]
    return points


def _remove_small_components(
    binary: np.ndarray,
    *,
    min_area: int,
    connectivity: Connectivity,
    keep_largest: bool,
) -> tuple[np.ndarray, int, int]:
    cv2_result = _remove_small_components_with_cv2(
        binary,
        min_area=min_area,
        connectivity=connectivity,
        keep_largest=keep_largest,
    )
    if cv2_result is not None:
        return cv2_result

    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: list[Sequence[tuple[int, int]]] = []

    neighbors = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 4
        else [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    )

    for start_y, start_x in zip(*np.where(binary & ~visited)):
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        component: list[tuple[int, int]] = []

        while queue:
            y, x = queue.popleft()
            component.append((y, x))

            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not binary[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

        if len(component) >= min_area:
            components.append(component)

    if keep_largest and components:
        components = [max(components, key=len)]

    denoised = np.zeros_like(binary, dtype=bool)
    for component in components:
        ys, xs = zip(*component)
        denoised[np.asarray(ys), np.asarray(xs)] = True

    original_component_count = _count_components(binary, connectivity)
    kept_components = len(components)
    removed_components = max(0, original_component_count - kept_components)
    return denoised, removed_components, kept_components


def _remove_small_components_with_cv2(
    binary: np.ndarray,
    *,
    min_area: int,
    connectivity: Connectivity,
    keep_largest: bool,
) -> tuple[np.ndarray, int, int] | None:
    try:
        import cv2  # type: ignore
    except ImportError:
        return None

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8),
        connectivity=connectivity,
    )
    if num_labels <= 1:
        return np.zeros_like(binary, dtype=bool), 0, 0

    component_ids = [
        label
        for label in range(1, num_labels)
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area
    ]
    if keep_largest and component_ids:
        component_ids = [
            max(component_ids, key=lambda label: int(stats[label, cv2.CC_STAT_AREA]))
        ]

    denoised = np.isin(labels, component_ids)
    kept_components = len(component_ids)
    removed_components = max(0, (num_labels - 1) - kept_components)
    return denoised, removed_components, kept_components


def _count_components(binary: np.ndarray, connectivity: Connectivity) -> int:
    # Reuse the remover with min_area=1 without keep_largest, but avoid recursion.
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    neighbors = (
        [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 4
        else [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    )

    for start_y, start_x in zip(*np.where(binary & ~visited)):
        if visited[start_y, start_x]:
            continue
        count += 1
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or not binary[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))
    return count
