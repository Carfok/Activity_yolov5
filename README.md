# Activity YOLOv5 — Pedestrian & Crosswalk Detection

This repository contains configuration, scripts, and notebooks to train a
**YOLOv5** model to detect:

| Class ID | Name | Description |
|----------|------|-------------|
| `0` | `person` | Pedestrian (one bounding box per person) |
| `1` | `crosswalk` | Pedestrian crossing area (**one** bounding box per crossing) |

The training data combines:
- **COCO 2017** (filtered to `person` only) — provides general pedestrian appearances.
- **Custom CCTV dataset** (≥ 800 images) — contains both `person` and `crosswalk` labels
  from real surveillance footage.

---

## Repository structure

```
Activity_yolov5/
├── data/
│   └── cctv_crosswalk.yaml          # YOLOv5 dataset config (person + crosswalk)
├── scripts/
│   └── convert_coco_person_to_yolo.py  # COCO → YOLO conversion (person only)
├── Training_CCTV_Crosswalk.ipynb    # Colab notebook — train person + crosswalk
├── Training_Solar_Panels.ipynb      # Legacy notebook — solar panel damage detection
├── hyp.scratch_custom.yaml          # Custom hyperparameters
└── datasets/
    └── cctv_crosswalk/              # (not committed — create locally)
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/
```

> **Note:** The `datasets/` directory is **not** committed to the repository.
> You must create it locally by following the steps below.

---

## 1. Prepare the CCTV dataset

### Folder structure

Create the following directory tree before adding images:

```bash
mkdir -p datasets/cctv_crosswalk/images/{train,val,test}
mkdir -p datasets/cctv_crosswalk/labels/{train,val,test}
```

### Recommended data split

| Split | Share | Minimum images |
|-------|-------|----------------|
| `train` | 70 % | 560 |
| `val` | 20 % | 160 |
| `test` | 10 % | 80 |

Aim for **≥ 800 images total** in the CCTV portion. Each image must have a
corresponding label file with the **same stem** in the matching `labels/`
subdirectory.

### YOLO label format

Each label file (`<image_stem>.txt`) contains one detection per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalised to `[0, 1]`** relative to the image dimensions.

Example — an image 1280 × 720 containing one person and one crosswalk:

```
0 0.512 0.640 0.120 0.350
1 0.500 0.900 0.800 0.180
```

### Labeling guidelines

#### `person` (class 0)
- Draw a tight bounding box from the top of the head to the feet.
- Label every visible pedestrian individually.
- Partially occluded persons should still be labelled if at least 30 % visible.

#### `crosswalk` (class 1)
- Draw **exactly one** bounding box that covers the entire painted crossing area.
- Include the full striped rectangle, edge to edge.
- Do **not** draw separate boxes for individual stripes.
- If a crossing exits the frame, label only the visible portion.

---

## 2. Download COCO 2017 and filter to `person`

### Download

```bash
# Images (large — ~19 GB train, ~1 GB val)
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Annotations (~241 MB)
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip
unzip train2017.zip -d coco/images/
unzip val2017.zip   -d coco/images/
unzip annotations_trainval2017.zip -d coco/
```

Expected layout after unzipping:

```
coco/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── images/
    ├── train2017/
    └── val2017/
```

### Convert to YOLO (person only)

```bash
python scripts/convert_coco_person_to_yolo.py \
    --coco_dir /path/to/coco \
    --output_dir datasets/cctv_crosswalk \
    --splits train val \
    --no_empty
```

| Flag | Default | Description |
|------|---------|-------------|
| `--coco_dir` | *(required)* | Root directory of the downloaded COCO dataset |
| `--output_dir` | `datasets/cctv_crosswalk` | Where to write images + labels |
| `--splits` | `train val` | Which splits to process |
| `--copy_images` | off | Copy images (uses disk space); default is to symlink |
| `--no_empty` | off | Skip COCO images that have no person annotation |

After conversion, merge your CCTV images/labels into the same output directory.
The `crosswalk` labels (class `1`) come exclusively from the CCTV dataset.

---

## 3. Train

### Prerequisites

```bash
git clone https://github.com/ultralytics/yolov5 yolov5
pip install -r yolov5/requirements.txt
```

### Training command

```bash
python yolov5/train.py \
    --data    data/cctv_crosswalk.yaml \
    --weights yolov5s.pt \
    --img     640 \
    --batch   16 \
    --epochs  100 \
    --project runs/train \
    --name    cctv_crosswalk
```

#### Recommended flags by hardware

| Setting | T4 / 16 GB VRAM | RTX 3090 / 24 GB | CPU only |
|---------|----------------|-----------------|---------|
| `--img` | `640` | `640` | `416` |
| `--batch` | `16` | `32` | `4` |
| `--epochs` | `100` | `100` | `50` |
| `--weights` | `yolov5s.pt` | `yolov5m.pt` | `yolov5s.pt` |
| `--cache` | `ram` | `ram` | *(omit)* |
| `--workers` | `8` | `8` | `2` |

#### Resume a training run

```bash
python yolov5/train.py --resume runs/train/cctv_crosswalk/weights/last.pt
```

---

## 4. Validate

```bash
python yolov5/val.py \
    --data    data/cctv_crosswalk.yaml \
    --weights runs/train/cctv_crosswalk/weights/best.pt \
    --img     640 \
    --task    val \
    --verbose
```

---

## 5. Detect / inference

```bash
# On a single image
python yolov5/detect.py \
    --weights runs/train/cctv_crosswalk/weights/best.pt \
    --source  path/to/image.jpg \
    --data    data/cctv_crosswalk.yaml \
    --img     640

# On a video
python yolov5/detect.py \
    --weights runs/train/cctv_crosswalk/weights/best.pt \
    --source  path/to/video.mp4 \
    --data    data/cctv_crosswalk.yaml \
    --img     640

# On a webcam (device 0)
python yolov5/detect.py \
    --weights runs/train/cctv_crosswalk/weights/best.pt \
    --source  0 \
    --data    data/cctv_crosswalk.yaml \
    --img     640
```

Results are saved to `runs/detect/`.

---

## 6. Google Colab

Open **`Training_CCTV_Crosswalk.ipynb`** in Google Colab for a step-by-step
walkthrough that covers:

1. GPU check
2. YOLOv5 installation
3. Dataset conversion (COCO person) and CCTV data upload
4. Training with `yolov5s.pt`
5. Validation and results visualisation
6. Downloading the trained weights

---

## Legacy: Solar Panel Damage Detection

The original training notebook (`Training_Solar_Panels.ipynb`) and dataset
configuration (`personalizado.yaml`) are preserved for reference. They cover
detection of three classes: `Hotspots_BurnedCells`, `Broken_Cracks`, and
`Soiling`.
