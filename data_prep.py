"""RAM-H1200 erosion-mask -> YOLO detection-label conversion sketch."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_SUFFIXES = IMAGE_SUFFIXES
SPLITS = ("train", "val", "test")
MIN_COMPONENT_AREA = 16


def find_mask(mask_dir: Path, image_stem: str) -> Path:
    """Pair files by stem; adapt this function if the dataset uses another naming rule."""
    candidates = [p for p in mask_dir.glob(f"{image_stem}.*")
                  if p.suffix.lower() in MASK_SUFFIXES]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one mask for {image_stem!r} under {mask_dir}, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def mask_to_yolo_boxes(mask: np.ndarray,
                       min_area: int = MIN_COMPONENT_AREA) -> list[tuple[float, ...]]:
    """Return normalized (class, cx, cy, width, height) boxes for 8-connected regions."""
    if mask.ndim == 3:
        # RAM-H1200 masks are expected to encode erosion as nonzero pixels.
        # If the source uses a dedicated palette/class ID, select that ID here.
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes: list[tuple[float, ...]] = []
    for component_id in range(1, count):  # id 0 is the background
        x, y, box_w, box_h, area = stats[component_id].tolist()
        if area < min_area:
            continue

        # cv2 stats use half-open extents [x, x + box_w), [y, y + box_h).
        cx = (x + box_w / 2.0) / width
        cy = (y + box_h / 2.0) / height
        nw = box_w / width
        nh = box_h / height
        boxes.append((0.0, cx, cy, nw, nh))
    return boxes


def convert_split(source_root: Path, output_root: Path, split: str) -> None:
    image_dir = source_root / split / "images"
    mask_dir = source_root / split / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {image_dir} and {mask_dir}; adjust the directory mapping "
            "to match the official RAM-H1200 release."
        )

    out_images = output_root / "images" / split
    out_labels = output_root / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in images:
        mask_path = find_mask(mask_dir, image_path.stem)
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f"Could not read image/mask pair: {image_path}, {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask sizes differ for {image_path.name}: "
                f"{image.shape[:2]} vs {mask.shape[:2]}"
            )

        labels = mask_to_yolo_boxes(mask)
        label_path = out_labels / f"{image_path.stem}.txt"
        with label_path.open("w", encoding="utf-8", newline="\n") as handle:
            for class_id, cx, cy, box_w, box_h in labels:
                handle.write(
                    f"{int(class_id)} {cx:.8f} {cy:.8f} {box_w:.8f} {box_h:.8f}\n"
                )
        shutil.copy2(image_path, out_images / image_path.name)


def write_dataset_yaml(output_root: Path) -> None:
    # Relative paths keep the generated dataset portable when the folder is moved.
    content = "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bone_erosion\n"
    (output_root / "dataset.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="RAM-H1200 root with train/val/test/{images,masks}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        convert_split(args.source, args.output, split)
    write_dataset_yaml(args.output)


if __name__ == "__main__":
    main()
