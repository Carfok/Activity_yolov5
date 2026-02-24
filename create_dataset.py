import os
from collections import Counter
from pathlib import Path

# Class names for solar panel damage detection
CLASS_NAMES = {
    0: "Hotspots/Burned Cells",
    1: "Broken/Cracks",
    2: "Soiling",
}

# Resolve base_path: support running from repo root or any working directory
REPO_ROOT = Path(__file__).parent
base_path = REPO_ROOT / 'data'
source_img_train = base_path / 'images' / 'train'
source_img_val = base_path / 'images' / 'val'
source_lbl_train = base_path / 'labels' / 'train'
source_lbl_val = base_path / 'labels' / 'val'


def count_class_distribution(lbl_dir):
    """Return a Counter of class IDs found across all label files in lbl_dir."""
    distribution = Counter()
    for lbl_file in lbl_dir.glob('*.txt'):
        with open(lbl_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    distribution[class_id] += 1
    return distribution


def verify_dataset(img_dir, lbl_dir, split_name):
    img_files = list(img_dir.glob('*'))
    lbl_files = list(lbl_dir.glob('*.txt'))

    img_stems = {f.stem for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    lbl_stems = {f.stem for f in lbl_files}

    matches = img_stems.intersection(lbl_stems)
    mismatches_img = img_stems - lbl_stems
    mismatches_lbl = lbl_stems - img_stems

    print(f"--- Split: {split_name} ---")
    print(f"  Images:         {len(img_stems)}")
    print(f"  Labels:         {len(lbl_stems)}")
    print(f"  Matching pairs: {len(matches)}")

    if mismatches_img:
        print(f"  Images without labels: {len(mismatches_img)}")
    if mismatches_lbl:
        print(f"  Labels without images: {len(mismatches_lbl)}")

    # Class distribution
    if lbl_dir.exists():
        dist = count_class_distribution(lbl_dir)
        print(f"  Class distribution:")
        for class_id in sorted(dist.keys()):
            name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            print(f"    Class {class_id} ({name}): {dist[class_id]} instances")

    return len(matches)


train_count = verify_dataset(source_img_train, source_lbl_train, "train")
val_count = verify_dataset(source_img_val, source_lbl_val, "val")

print(f"\nTotal matching pairs: {train_count + val_count}")

# Update dataset.yaml with portable relative paths
dataset_yaml_path = REPO_ROOT / 'dataset.yaml'

yaml_content = f"""# Dataset configuration for Solar Panel Damage Detection
train: {source_img_train}
val: {source_img_val}

# Number of classes
nc: 3

# Class names
names: ['Hotspots_BurnedCells', 'Broken_Cracks', 'Soiling']
"""

with open(dataset_yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\nUpdated {dataset_yaml_path}")
