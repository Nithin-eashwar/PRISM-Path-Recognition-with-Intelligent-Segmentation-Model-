# PRISM - Path Recognition with Intelligent Segmentation Model

Real-time drivable-space segmentation on nuScenes `v1.0-mini`, built from scratch in PyTorch.

## Project Overview

PRISM performs binary segmentation on front-camera driving images:
- `1` = drivable
- `0` = non-drivable

Key capabilities included in this repo:
- A lightweight student model (`LiteSegNet`) for deployment
- A larger teacher model (`LiteSegTeacher`) for knowledge distillation
- Custom PRISM loss (`PRISMLossV2`) to suppress false positives (sky/buildings -> road)
- Boundary refinement and test-time augmentation (TTA) for sharper masks
- Inference utilities: ONNX export, FPS benchmarking, demo video generation
- Optional vehicle suppression using YOLOv8n during inference

## Model Architecture

### Core Pipeline (student and teacher share the same design)

Input (`3 x 256 x 448`) ->
1. **CoordConv stem** for explicit (x, y) spatial priors
2. **MobileNetV2-style encoder** with inverted residual blocks
3. **RAU (Reflection Attention Unit)** to handle reflective road/puddle regions
4. **Lightweight ASPP** (dilations 6/12/18 + global pooling)
5. **U-Net decoder** with skip connections + SE attention
6. **Segmentation head** -> raw logits (sigmoid applied externally)
7. **Auxiliary boundary head** (training-only) for edge supervision

### Student vs Teacher (from repo metrics)

| Model | Parameters | Purpose |
|---|---:|---|
| LiteSegNet (student) | ~1.86M | Fast inference / deployment |
| LiteSegTeacher | ~1.97M | Teacher for distillation |

## Dataset Used

### Source

- **nuScenes `v1.0-mini`**
- 10 scenes, 404 CAM_FRONT keyframes
- Original resolution: `1600 x 900`
- Training/inference resolution: `448 x 256`

### Required Layout (under `--dataroot`)

```
v1.0-mini/
  scene.json
  sample.json
  sample_data.json
  calibrated_sensor.json
  ego_pose.json
  sensor.json
  map.json
samples/
  CAM_FRONT/   # front camera images (.jpg)
maps/          # nuScenes bitmap semantic priors (referenced by map.json)
```

### Labels / Masks

Drivable masks are generated from nuScenes semantic-prior bitmaps:
- `generate_masks.py` auto-calibrates bitmap-to-world alignment using ego poses
- Masks are projected into camera space and saved as PNGs in `masks/`
- `masks/file_mapping.json` stores `(image_rel_path, mask_filename)` pairs

Train/val split is **scene-based** to prevent leakage (default = last 2 scenes for validation).

## Setup & Installation Instructions

Run from repository root:

```powershell
cd "path\to\PRISM-Path-Recognition-with-Intelligent-Segmentation-Model-"
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional dependency (only if you use `--suppress_vehicles` in inference):

```powershell
pip install ultralytics
```

## How to Run the Code

### 1) Generate Masks

```bash
python generate_masks.py --dataroot ./ --output_dir masks --visualize 10
```

### 2) Training (recommended distillation-first sequence)

1. Train teacher
```bash
python train.py --dataroot ./ --mask_dir masks --epochs 50 --batch_size 16 --lr 1e-3 --train_teacher --save_name teacher_best.pth
```

2. Distill student from teacher
```bash
python train.py --dataroot ./ --mask_dir masks --epochs 50 --batch_size 16 --lr 1e-3 --distill --teacher_weights output/teacher_best.pth --save_name best_model_distilled.pth
```

3. Optional baseline student (no distillation)
```bash
python train.py --dataroot ./ --mask_dir masks --epochs 50 --batch_size 16 --lr 1e-3 --save_name best_model_baseline.pth
```

Notes:
- Default loss is `PRISMLossV2` (focal + tversky + multi-scale boundary + spatial prior).
- Use `--loss combo` or `--loss boundary` for legacy alternatives.
- `--grad_accum` lets you emulate larger batches on smaller GPUs.

### 3) Evaluate

```bash
python evaluate.py --weights output/best_model_distilled.pth --dataroot ./ --mask_dir masks --use_tta --use_boundary_refinement
```

This produces:
- `eval_output/metrics.json`
- `eval_output/confusion_matrix.png`
- `eval_output/visualizations/` (overlay samples)

### 4) Inference

Single image:
```bash
python inference.py --image path/to/image.jpg --weights output/best_model_distilled.pth --refine --tta
```

Optional vehicle suppression (YOLOv8n):
```bash
python inference.py --image path/to/image.jpg --weights output/best_model_distilled.pth --refine --tta --suppress_vehicles
```

ONNX export + benchmark:
```bash
python inference.py --weights output/best_model_distilled.pth --export_onnx --benchmark --quantize
```

Demo video (from scene sequence):
```bash
python inference.py --weights output/best_model_distilled.pth --demo_video --dataroot ./
```

### 5) TensorBoard

```bash
tensorboard --logdir runs
```

## Example Outputs / Results

Below are example metrics pulled from repo artifacts (TTA + boundary refinement enabled):

### Example A (from `visualizations/metrics.json`)

| Metric | Value |
|---|---:|
| mIoU overall | 0.8841 |
| mIoU drivable | 0.8911 |
| FPS | 145.8 |
| Params | 1,863,537 |

### Example B (from `eval_distilled/metrics.json`)

| Metric | Value |
|---|---:|
| mIoU overall | 0.8037 |
| mIoU drivable | 0.8228 |
| FPS | 32.7 |
| Params | 1,866,793 |

### Example C (from `eval_output/metrics.json`)

| Metric | Value |
|---|---:|
| mIoU overall | 0.8789 |
| mIoU drivable | 0.8521 |
| FPS | 54.8 |
| Params | 1,967,179 |

### Sample Visual Outputs

- `visualizations/eval_sample_00.png`
- `visualizations/eval_sample_01.png`
- `eval_distilled/visualizations/eval_sample_00.png`
- `eval_output/visualizations/eval_sample_00.png`

You can generate fresh outputs using `evaluate.py` and `inference.py` as shown above.
