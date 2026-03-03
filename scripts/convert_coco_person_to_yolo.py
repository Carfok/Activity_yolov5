#!/usr/bin/env python3
"""
convert_coco_person_to_yolo.py
-------------------------------
Convert COCO 2017 annotations to YOLO format, keeping ONLY the `person` class.

The script reads the official COCO JSON annotation files and produces:
  - One .txt label file per image (YOLO format: class x_c y_c w h, normalised)
  - Organised directory structure ready for `data/cctv_crosswalk.yaml`

Output structure
----------------
  <output_dir>/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/

Usage
-----
1. Download COCO 2017:
     wget http://images.cocodataset.org/zips/train2017.zip
     wget http://images.cocodataset.org/zips/val2017.zip
     wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
   Unzip so you have:
     coco/
       annotations/instances_train2017.json
       annotations/instances_val2017.json
       images/train2017/
       images/val2017/

2. Run this script:
     python scripts/convert_coco_person_to_yolo.py \\
         --coco_dir /path/to/coco \\
         --output_dir datasets/cctv_crosswalk

   Optional flags:
     --splits train val          (default: train val)
     --copy_images               copy matching images to output_dir (default: symlink)
     --no_empty                  skip images that have no person annotation
"""

import argparse
import json
import os
import shutil
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_coco_json(json_path: Path) -> dict:
    """Load a COCO-format JSON file and return its contents."""
    with open(json_path, "r") as f:
        return json.load(f)


def get_person_category_id(coco_data: dict) -> int:
    """Return the COCO category id for 'person' (raises if not found)."""
    for cat in coco_data.get("categories", []):
        if cat["name"] == "person":
            return cat["id"]
    raise ValueError("Category 'person' not found in the provided COCO JSON.")


def coco_bbox_to_yolo(bbox: list, img_w: int, img_h: int) -> tuple:
    """
    Convert COCO bbox [x_min, y_min, width, height] to
    YOLO [x_center, y_center, width, height] (all values normalised to [0, 1]).
    """
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    # Clamp to [0, 1] to handle minor annotation offsets
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    return x_center, y_center, w_norm, h_norm


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_split(
    coco_json_path: Path,
    images_src_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    copy_images: bool,
    skip_empty: bool,
) -> dict:
    """
    Process one COCO split (train or val).

    Returns a summary dict with conversion statistics.
    """
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Loading {coco_json_path} …")
    coco = load_coco_json(coco_json_path)

    person_id = get_person_category_id(coco)
    print(f"[INFO] COCO 'person' category id = {person_id}")

    # Build lookup: image_id → image metadata
    id_to_image = {img["id"]: img for img in coco["images"]}

    # Collect person annotations grouped by image_id
    person_anns: dict[int, list] = {}
    skipped_crowd = 0
    for ann in coco.get("annotations", []):
        if ann["category_id"] != person_id:
            continue
        if ann.get("iscrowd", 0):
            skipped_crowd += 1
            continue
        img_id = ann["image_id"]
        person_anns.setdefault(img_id, []).append(ann)

    # Process images
    images_written = 0
    labels_written = 0
    images_skipped_empty = 0

    for img_id, img_info in id_to_image.items():
        anns = person_anns.get(img_id, [])

        if skip_empty and not anns:
            images_skipped_empty += 1
            continue

        filename = img_info["file_name"]           # e.g. "000000001234.jpg"
        src_img = images_src_dir / filename
        dst_img = output_images_dir / filename
        dst_lbl = output_labels_dir / (Path(filename).stem + ".txt")

        # Copy / symlink image
        if src_img.exists():
            if copy_images:
                if not dst_img.exists():
                    shutil.copy2(src_img, dst_img)
            else:
                if not dst_img.exists():
                    dst_img.symlink_to(src_img.resolve())
            images_written += 1
        else:
            # Image file missing — write label anyway if there are annotations
            pass

        # Write label file (YOLO format, class 0 = person)
        img_w = img_info["width"]
        img_h = img_info["height"]
        lines = []
        for ann in anns:
            x_c, y_c, w_n, h_n = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        labels_written += 1

    summary = {
        "total_images": len(id_to_image),
        "images_written": images_written,
        "labels_written": labels_written,
        "images_skipped_empty": images_skipped_empty,
        "person_annotations": sum(len(v) for v in person_anns.values()),
        "crowd_annotations_skipped": skipped_crowd,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert COCO 2017 → YOLO format (person class only)."
    )
    parser.add_argument(
        "--coco_dir",
        type=Path,
        required=True,
        help="Root directory of the COCO dataset (must contain annotations/ and images/ subdirs).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("datasets/cctv_crosswalk"),
        help="Output root directory (default: datasets/cctv_crosswalk).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Which splits to process (default: train val).",
    )
    parser.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy images to output_dir instead of creating symlinks (uses more disk space).",
    )
    parser.add_argument(
        "--no_empty",
        action="store_true",
        help="Skip images that contain no person annotations.",
    )
    args = parser.parse_args()

    coco_dir: Path = args.coco_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    print(f"[INFO] COCO source : {coco_dir}")
    print(f"[INFO] Output dir  : {output_dir}")
    print(f"[INFO] Splits      : {args.splits}")
    print(f"[INFO] Copy images : {args.copy_images}")
    print(f"[INFO] Skip empty  : {args.no_empty}")

    # Map split name → COCO JSON name and images subfolder name
    split_map = {
        "train": ("instances_train2017.json", "train2017"),
        "val":   ("instances_val2017.json",   "val2017"),
    }

    for split in args.splits:
        json_name, img_folder = split_map[split]
        coco_json_path = coco_dir / "annotations" / json_name
        images_src_dir = coco_dir / "images" / img_folder

        if not coco_json_path.exists():
            print(f"[WARNING] {coco_json_path} not found — skipping split '{split}'.")
            continue

        out_images = output_dir / "images" / split
        out_labels = output_dir / "labels" / split

        summary = convert_split(
            coco_json_path=coco_json_path,
            images_src_dir=images_src_dir,
            output_images_dir=out_images,
            output_labels_dir=out_labels,
            copy_images=args.copy_images,
            skip_empty=args.no_empty,
        )

        print(f"\n[RESULT] Split '{split}':")
        print(f"  Total COCO images      : {summary['total_images']}")
        print(f"  Images written         : {summary['images_written']}")
        print(f"  Label files written    : {summary['labels_written']}")
        print(f"  Person annotations     : {summary['person_annotations']}")
        print(f"  Crowd annotations skip : {summary['crowd_annotations_skipped']}")
        if summary["images_skipped_empty"]:
            print(f"  Images skipped (empty) : {summary['images_skipped_empty']}")

    print("\n[DONE] Conversion complete.")
    print(f"       Labels are in  : {output_dir / 'labels'}")
    print(f"       Images are in  : {output_dir / 'images'}")
    print(
        "\nNext step: add your CCTV crosswalk images + labels (class 1) to the same"
        " directories, then train with:\n"
        "  python yolov5/train.py \\\n"
        "      --data data/cctv_crosswalk.yaml \\\n"
        "      --weights yolov5s.pt \\\n"
        "      --img 640 --batch 16 --epochs 100"
    )


if __name__ == "__main__":
    main()
