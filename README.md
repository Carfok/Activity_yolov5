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
│   └── cctv_crosswalk/              # Your 880 COCO person images
│       ├── images/{train,val}
│       └── labels/{train,val}
├── data/cctv_crosswalk.yaml         # YOLOv5 dataset config
├── Training_CCTV_Crosswalk.ipynb    # Colab notebook — train person + crosswalk
└── yolov5s.pt                       # Pretrained weights
```

---

## 1. Dataset Status

The repository now contains **880 images** of people from the COCO dataset, specifically filtered and converted to YOLOv5 format for pedestrian detection tasks.

- **Split**: 80% Train (704 images), 20% Val (176 images).
- **Class**: `0 — person` (pedestrian).

---

## 2. Train

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

---

## 3. Validate

```bash
python yolov5/val.py \
    --data    data/cctv_crosswalk.yaml \
    --weights runs/train/cctv_crosswalk/weights/best.pt \
    --img     640 \
    --task    val \
    --verbose
```

---

## 4. Detect / inference

```bash
# On a single image
python yolov5/detect.py \
    --weights runs/train/cctv_crosswalk/weights/best.pt \
    --source  path/to/image.jpg \
    --data    data/cctv_crosswalk.yaml \
    --img     640
```

---

## 5. Google Colab

Use **`Training_CCTV_Crosswalk.ipynb`** for training in the cloud.
