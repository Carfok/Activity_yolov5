import os
from pathlib import Path

# Paths to the existing structure
base_path = Path(r'c:\Users\Techie2\Downloads\Actividad_1\data\data')
source_img_train = base_path / 'images' / 'train'
source_img_val = base_path / 'images' / 'val'
source_lbl_train = base_path / 'labels' / 'train'
source_lbl_val = base_path / 'labels' / 'val'

def verify_dataset(img_dir, lbl_dir, split_name):
    img_files = list(img_dir.glob('*'))
    lbl_files = list(lbl_dir.glob('*.txt'))
    
    img_stems = {f.stem for f in img_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']}
    lbl_stems = {f.stem for f in lbl_files}
    
    matches = img_stems.intersection(lbl_stems)
    mismatches_img = img_stems - lbl_stems
    mismatches_lbl = lbl_stems - img_stems
    
    print(f"--- Split: {split_name} ---")
    print(f"Images: {len(img_stems)}")
    print(f"Labels: {len(lbl_stems)}")
    print(f"Matching pairs: {len(matches)}")
    
    if mismatches_img:
        print(f"Images without labels: {len(mismatches_img)}")
    if mismatches_lbl:
        print(f"Labels without images: {len(mismatches_lbl)}")
    return len(matches)

train_count = verify_dataset(source_img_train, source_lbl_train, "train")
val_count = verify_dataset(source_img_val, source_lbl_val, "val")

print(f"\nTotal pairs: {train_count + val_count}")

# Path to the dataset configuration file
dataset_yaml_path = Path(r'c:\Users\Techie2\Downloads\Actividad_1\dataset.yaml')

# Content of the dataset.yaml file
yaml_content = f"""# Dataset configuration for Solar Panel Damage Detection
train: {source_img_train}
val: {source_img_val}

# Number of classes
nc: 3

# Class names
names: ['class0', 'class1', 'class2']
"""

with open(dataset_yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"Updated {dataset_yaml_path}")
