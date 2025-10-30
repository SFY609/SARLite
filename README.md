# SARLite: Lightweight Extensions for YOLO11 in SAR Object Detection

## 🛰️ Purpose

This repository provides the **SARLite module implementation** — a set of lightweight extensions to **YOLO11** designed for **Synthetic Aperture Radar (SAR)** object detection.  
The included code focuses on **custom modules** (enhanced HGNet backbone, lightweight block, and DySample upsampling) and **YAML configuration files** defining the model structure.  

The core YOLO framework is **not included here**; please refer to the official YOLO11 repository for full training and inference workflows:
> 🔗 [Ultralytics YOLO11 GitHub](https://github.com/ultralytics/ultralytics)

---

## ⚙️ Summary of Modifications

| Module | Description | Purpose |
|---------|--------------|----------|
| **HGNet Backbone** | Hybrid-group convolution backbone for improved multi-scale feature extraction in SAR imagery. | Stronger SAR-specific representation |
| **Lightweight Block** | Structural simplification to reduce computational cost while maintaining performance. | Faster inference, fewer parameters |
| **DySample Upsample** | Dynamic upsampling layer based on linear + pixel shuffle operations. | Better fine-grained spatial recovery |

These components collectively form the **SARLite** model variant used in our paper:
> *"Lightweight SAR Object Detection with Enhanced HGNet Backbone and Dynamic Upsampling"*  
> (Under review)

---

## 📁 Repository Contents

```
SARLite/
│
├── SARLite.py/ # Implementation of custom modules
│
├── yolo11_sarlite.yaml/ # Model configuration file
│
└── README.md
```
---

## 📁 Dataset

All experiments in the accompanying paper were conducted using SARDet-100K, a large-scale benchmark dataset for SAR object detection.

> 🔗 [SARDet_100K GitHub](https://github.com/zcablii/SARDet_100K)
