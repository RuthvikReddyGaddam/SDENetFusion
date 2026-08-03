#!/usr/bin/env python


import base64
import io
import json
import zlib

import numpy as np
from PIL import Image


def load_mask_numpy(annotation_path):
    """
    Reads a Supervisely .bmp.json annotation and creates one combined
    binary gland mask.

    Returns:
        NumPy array with shape [H, W]
        dtype: uint8

        Background = 0
        Gland      = 1
    """

    with open(annotation_path, "r", encoding="utf-8") as file:
        annotation = json.load(file)

    image_height = int(annotation["size"]["height"])
    image_width = int(annotation["size"]["width"])

    full_mask = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    for obj in annotation.get("objects", []):
        if obj.get("classTitle") != "gland":
            continue

        if obj.get("geometryType") != "bitmap":
            continue

        bitmap = obj["bitmap"]

        # Supervisely origin format: [x, y]
        origin_x, origin_y = map(int, bitmap["origin"])

        compressed_data = base64.b64decode(bitmap["data"])
        png_data = zlib.decompress(compressed_data)

        local_mask = np.asarray(
            Image.open(io.BytesIO(png_data)).convert("L"),
            dtype=np.uint8,
        )

        local_mask = (local_mask > 0).astype(np.uint8)

        local_height, local_width = local_mask.shape

        start_x = max(origin_x, 0)
        start_y = max(origin_y, 0)

        end_x = min(
            origin_x + local_width,
            image_width,
        )

        end_y = min(
            origin_y + local_height,
            image_height,
        )

        if end_x <= start_x or end_y <= start_y:
            continue

        local_start_x = start_x - origin_x
        local_start_y = start_y - origin_y

        valid_width = end_x - start_x
        valid_height = end_y - start_y

        existing_region = full_mask[
            start_y:end_y,
            start_x:end_x,
        ]

        local_region = local_mask[
            local_start_y:local_start_y + valid_height,
            local_start_x:local_start_x + valid_width,
        ]

        full_mask[start_y:end_y, start_x:end_x] = np.maximum(
            existing_region,
            local_region,
        )

    return full_mask


import csv
from pathlib import Path

import numpy as np
from PIL import Image


def make_pair(value):
    if isinstance(value, int):
        return value, value

    return int(value[0]), int(value[1])


def generate_positions(
    image_size,
    patch_size,
    stride,
):
    """
    Generates sliding-window positions and adds a final border-aligned
    position when needed.
    """

    maximum_position = image_size - patch_size

    positions = list(
        range(
            0,
            maximum_position + 1,
            stride,
        )
    )

    if positions[-1] != maximum_position:
        positions.append(maximum_position)

    return positions


def pad_image_and_mask(
    image,
    mask,
    patch_height,
    patch_width,
):
    """
    Pads only when an image dimension is smaller than the patch size.

    Image padding: white, value 255
    Mask padding:  black, value 0
    """

    image_height, image_width = image.shape[:2]

    padded_height = max(image_height, patch_height)
    padded_width = max(image_width, patch_width)

    if (
        padded_height == image_height
        and padded_width == image_width
    ):
        return image, mask

    padded_image = np.full(
        (padded_height, padded_width, 3),
        fill_value=255,
        dtype=np.uint8,
    )

    padded_mask = np.zeros(
        (padded_height, padded_width),
        dtype=np.uint8,
    )

    padded_image[
        :image_height,
        :image_width,
    ] = image

    padded_mask[
        :image_height,
        :image_width,
    ] = mask

    return padded_image, padded_mask


def preprocess_glaS_patches(
    image_dir,
    annotation_dir,
    output_dir,
    patch_size=256,
    stride=128,
):
    """
    Converts full GlaS images and Supervisely JSON annotations into
    precomputed NumPy patches.

    Output structure:

        output_dir/
            images/
                train_1_patch_0000.npy
                ...
            masks/
                train_1_patch_0000.npy
                ...
            manifest.csv
    """

    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)

    output_image_dir = output_dir / "images"
    output_mask_dir = output_dir / "masks"

    output_image_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_height, patch_width = make_pair(patch_size)
    stride_height, stride_width = make_pair(stride)

    image_paths = sorted(image_dir.glob("*.bmp"))

    manifest_rows = []
    total_patches = 0

    for image_path in image_paths:
        annotation_path = (
            annotation_dir
            / f"{image_path.name}.json"
        )

        image = np.asarray(
            Image.open(image_path).convert("RGB"),
            dtype=np.uint8,
        )

        mask = load_mask_numpy(annotation_path)

        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask size mismatch for {image_path.name}: "
                f"image={image.shape[:2]}, mask={mask.shape}"
            )

        original_height, original_width = image.shape[:2]

        image, mask = pad_image_and_mask(
            image=image,
            mask=mask,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        padded_height, padded_width = image.shape[:2]

        top_positions = generate_positions(
            image_size=padded_height,
            patch_size=patch_height,
            stride=stride_height,
        )

        left_positions = generate_positions(
            image_size=padded_width,
            patch_size=patch_width,
            stride=stride_width,
        )

        image_stem = image_path.stem
        image_patch_index = 0

        for top in top_positions:
            for left in left_positions:
                bottom = top + patch_height
                right = left + patch_width

                image_patch = image[
                    top:bottom,
                    left:right,
                    :,
                ]

                mask_patch = mask[
                    top:bottom,
                    left:right,
                ]

                patch_name = (
                    f"{image_stem}_patch_"
                    f"{image_patch_index:04d}"
                )

                image_output_path = (
                    output_image_dir
                    / f"{patch_name}.npy"
                )

                mask_output_path = (
                    output_mask_dir
                    / f"{patch_name}.npy"
                )

                np.save(
                    image_output_path,
                    image_patch,
                    allow_pickle=False,
                )

                np.save(
                    mask_output_path,
                    mask_patch,
                    allow_pickle=False,
                )

                foreground_pixels = int(mask_patch.sum())
                foreground_fraction = (
                    foreground_pixels
                    / mask_patch.size
                )

                manifest_rows.append({
                    "patch_name": patch_name,
                    "source_image": image_path.name,
                    "image_path": str(image_output_path),
                    "mask_path": str(mask_output_path),
                    "top": top,
                    "left": left,
                    "patch_height": patch_height,
                    "patch_width": patch_width,
                    "original_height": original_height,
                    "original_width": original_width,
                    "padded_height": padded_height,
                    "padded_width": padded_width,
                    "foreground_pixels": foreground_pixels,
                    "foreground_fraction": foreground_fraction,
                })

                image_patch_index += 1
                total_patches += 1

        print(
            f"{image_path.name}: "
            f"{image_patch_index} patches"
        )

    manifest_path = output_dir / "manifest.csv"

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nImages processed: {len(image_paths)}")
    print(f"Total patches: {total_patches}")
    print(f"Manifest: {manifest_path}")



if __name__ == "__main__":
    preprocess_glaS_patches(
    image_dir=(
        r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
        r"\SDENet-Fusion\Datasets\Glas\train\img"
    ),
    annotation_dir=(
        r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
        r"\SDENet-Fusion\Datasets\Glas\train\ann"
    ),
    output_dir=(
        r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
        r"\SDENet-Fusion\Datasets\Glas\precomputed"
        r"\train_256_stride_128"
    ),
    patch_size=256,
    stride=128,
    )


    preprocess_glaS_patches(
        image_dir=(
            r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
            r"\SDENet-Fusion\Datasets\Glas\test_a\img"
        ),
        annotation_dir=(
            r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
            r"\SDENet-Fusion\Datasets\Glas\test_a\ann"
        ),
        output_dir=(
            r"C:\Users\gadda\Documents\FeatureFusionMoNuSAC"
            r"\SDENet-Fusion\Datasets\Glas\precomputed"
            r"\test_a_256_stride_128"
        ),
        patch_size=256,
        stride=128,
    )