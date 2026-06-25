# YOLOv3 From Scratch

A clean, modular implementation of the **YOLOv3 (You Only Look Once)** object detection architecture built entirely from scratch using **PyTorch**.

This project focuses on understanding the internal mechanics of modern object detection systems by implementing the core components of YOLOv3, including the Darknet-53 backbone, multi-scale detection heads, anchor box predictions, loss calculations, and training pipeline.

---

## Overview

YOLOv3 is a single-stage object detection model that performs object localization and classification in a single forward pass, enabling real-time detection while maintaining strong accuracy.

This repository serves both as:

* A learning resource for understanding object detection architectures.
* A modular codebase for experimentation and research.

The implementation emphasizes readability, modularity, and traceability of the detection pipeline.

---

## Features

* Custom implementation of **Darknet-53 Backbone**
* Multi-scale object detection heads
* Anchor-box based predictions
* Custom YOLOv3 loss function
* Dataset preprocessing and augmentation pipeline
* Intersection over Union (IoU) calculations
* Non-Maximum Suppression (NMS)
* Training and evaluation utilities
* Modular and easy-to-extend architecture

---

## Repository Structure

```text
yoloV3_scratch/
│
├── config.py
├── dataset.py
├── loss.py
├── model.py
├── train.py
├── utils.py
├── README.md
└── requirements.txt
```

### File Descriptions

| File         | Description                                                                                     |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `config.py`  | Model hyperparameters, anchor boxes, image sizes, and training configurations                   |
| `dataset.py` | Dataset loading, preprocessing, target generation, and augmentation logic                       |
| `loss.py`    | YOLOv3 loss implementation including localization, objectness, and classification losses        |
| `model.py`   | Darknet-53 backbone and YOLOv3 detection head architecture                                      |
| `train.py`   | Training loop, optimizer setup, checkpoint handling, and validation                             |
| `utils.py`   | Utility functions including IoU computation, NMS, evaluation metrics, and visualization helpers |

---

## YOLOv3 Architecture

The network consists of:

### 1. Darknet-53 Backbone

A deep convolutional feature extractor composed of:

* Residual Blocks
* Batch Normalization
* Leaky ReLU Activations

Darknet-53 generates rich feature maps at multiple scales for object detection.

### 2. Detection Heads

YOLOv3 performs predictions at three different scales:

* Large Objects
* Medium Objects
* Small Objects

Each prediction head outputs:

* Bounding Box Coordinates
* Objectness Score
* Class Probabilities

---

## Anchor Boxes

YOLOv3 uses predefined anchor boxes to predict object dimensions efficiently.

The anchor configurations can be modified in:

```python
config.py
```

to suit custom datasets and object distributions.

---

## Loss Function

The loss implementation combines several components:

### Localization Loss

Measures the accuracy of predicted bounding box coordinates.

### Objectness Loss

Determines whether an object exists within a predicted region.

### No-Object Loss

Penalizes false-positive predictions.

### Classification Loss

Measures the accuracy of predicted object classes.

The modular implementation in `loss.py` makes each component easy to inspect and modify.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/yoloV3_scratch.git
cd yoloV3_scratch
```

### Install Dependencies

```bash
pip install torch torchvision numpy opencv-python
```

or

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The project expects datasets in YOLO format.

### Directory Structure

```text
dataset/
│
├── images/
│   ├── train/
│   └── val/
│
└── labels/
    ├── train/
    └── val/
```

### Label Format

Each label file contains:

```text
class_id x_center y_center width height
```

All coordinates should be normalized to the image dimensions.

---

## Training

Configure the dataset paths and hyperparameters inside:

```python
config.py
```

Then start training:

```bash
python train.py
```

Training options may include:

* Batch Size
* Learning Rate
* Number of Epochs
* Weight Decay
* Anchor Configurations

---

## Evaluation and Inference

Utilities for evaluation and inference are provided in:

```python
utils.py
```

Common functionality includes:

* IoU Calculation
* Non-Maximum Suppression (NMS)
* Bounding Box Conversion
* Prediction Filtering
* Detection Visualization

Example workflow:

```python
from utils import non_max_suppression

# Generate predictions
# Apply NMS
# Visualize detections
```

---

## Learning Objectives

This project is designed to help understand:

* Convolutional Neural Networks (CNNs)
* Residual Networks
* Feature Pyramid Detection
* Anchor-Based Object Detection
* Multi-Scale Predictions
* Bounding Box Regression
* IoU and NMS
* Detection Loss Functions
* End-to-End Object Detection Training

---

## Suggested Improvements

Potential future enhancements include:

* mAP (Mean Average Precision) evaluation
* Mixed Precision Training (AMP)
* Mosaic Data Augmentation
* Multi-GPU Training
* TensorBoard Logging
* ONNX Export
* COCO Dataset Support
* YOLOv4/YOLOv5 inspired improvements

---

## References

* YOLOv3: An Incremental Improvement (Redmon & Farhadi)
* Darknet Framework
* PyTorch Documentation
* COCO Dataset
* Pascal VOC Dataset

---

## Disclaimer

This implementation is intended primarily for educational and research purposes. The focus is on understanding and reproducing the core ideas behind YOLOv3 rather than matching highly optimized production-level implementations.

---

## License

This project is open-source and available under the MIT License.
