import os
import json
import argparse
import shutil
from tqdm import tqdm

def convert_coco_json(coco_dir, output_dir, split, copy_images=False, no_empty=True):
    # Path setup
    json_path = os.path.join(coco_dir, 'annotations', f'instances_{split}2017.json')
    img_dir = os.path.join(coco_dir, 'images', f'{split}2017')
    
    out_img_dir = os.path.join(output_dir, 'images', split)
    out_lbl_dir = os.path.join(output_dir, 'labels', split)
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Category ID for 'person' is 1 in COCO
    cat_person = 1
    
    # Map image IDs to annotations
    img_to_anns = {}
    for ann in data['annotations']:
        if ann['category_id'] == cat_person:
            img_id = ann['image_id']
            if img_id not in img_to_anns:
                img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

    print(f"Converting annotations for {split} split...")
    count = 0
    for img in tqdm(data['images']):
        img_id = img['id']
        file_name = img['file_name']
        anns = img_to_anns.get(img_id, [])
        
        if no_empty and not anns:
            continue

        # Convert to YOLO format
        yolo_anns = []
        img_w, img_h = img['width'], img['height']
        
        for ann in anns:
            # COCO: [x_min, y_min, width, height]
            x, y, w, h = ann['bbox']
            # YOLO: [class_id, x_center, y_center, width, height] (normalized)
            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h
            yolo_anns.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        # Save label file
        label_path = os.path.join(out_lbl_dir, file_name.replace('.jpg', '.txt'))
        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_anns))

        # Copy/Link image
        src_img = os.path.join(img_dir, file_name)
        dst_img = os.path.join(out_img_dir, file_name)
        
        if os.path.exists(src_img):
            if copy_images:
                shutil.copy(src_img, dst_img)
            else:
                # On Windows symlinks might fail without admin, so fallback to copy if needed
                try:
                    os.symlink(src_img, dst_img)
                except OSError:
                    shutil.copy(src_img, dst_img)
            count += 1

    print(f"Successfully processed {count} images for {split}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert COCO person annotations to YOLO format')
    parser.add_argument('--coco_dir', type=str, required=True, help='Path to COCO dataset root')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output YOLO dataset')
    parser.add_argument('--splits', type=str, nargs='+', default=['val'], help='Splits to process (train, val)')
    parser.add_argument('--copy_images', action='store_true', help='Copy images instead of symlinking')
    parser.add_argument('--no_empty', action='store_true', help='Skip images without persons')

    args = parser.parse_args()
    
    for split in args.splits:
        convert_coco_json(args.coco_dir, args.output_dir, split, args.copy_images, args.no_empty)
