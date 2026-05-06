# Mask To XY API

Convert a binary mask image into foreground pixel coordinates.

The API removes small connected noise components before returning coordinates.
Coordinates are returned as `[x, y]`, not `[row, column]`.

## Example

```python
from mask_to_xy_api import mask_to_xy_points

points = mask_to_xy_points(
    "output/sam31_verify/point_mask.png",
    threshold=127,
    min_area=32,
    connectivity=8,
)

print(points.shape)  # [N, 2]
print(points[:5])    # [[x, y], ...]
```

## Metadata

```python
from mask_to_xy_api import mask_to_xy_points

result = mask_to_xy_points(
    "mask.png",
    min_area=32,
    return_metadata=True,
)

print(result.points_xy)
print(result.original_foreground_pixels)
print(result.kept_foreground_pixels)
print(result.removed_components)
```

## Main API

- `mask_to_xy_points(mask, ...)`: returns an `np.ndarray` of shape `[N, 2]`.
- `mask_to_xy_list(mask, ...)`: returns `list[tuple[int, int]]`.
- `denoise_binary_mask(mask, ...)`: returns the cleaned bool mask.
- `load_binary_mask(mask, ...)`: loads and thresholds a mask.

Supported `mask` inputs:

- file path
- `PIL.Image.Image`
- `numpy.ndarray`

No command-line parser is provided; this folder is intended to be imported as a library.
