Edge AI on Kria KR260: ResNet-50 & YOLOX

Real-time deep learning on the AMD Kria KR260 with a custom DPU overlay. Runs ResNet-50 on PetaLinux and YOLOX-Nano on Ubuntu using DPU-PYNQ for robotic vision and edge inference.

Project Overview 💮

This project implements two edge AI workflows on the Kria KR260 platform using a custom B4096 DPU:

ResNet-50 classification on PetaLinux with Vitis AI 3.0

YOLOX-Nano object detection on Ubuntu with DPU-PYNQ 3.5 (Python)

Each pipeline demonstrates low-latency inference using shared DPU hardware.

System Architecture 🧩



Vivado DPU Design 🔧

The DPU overlay was created in Vivado 2022.2 with the B4096 architecture.

Unzip the kr260_dpu_vivado_screenshots.zip and include screenshots like:

![Block Design](images/vivado_block_design.png)
![DPU Config](images/dpu_config.png)

Workflow Summaries 📌

ResNet-50 on PetaLinux

C++ inference using VART APIs

Pretrained ResNet-50 from Model Zoo

Achieves ~0.2 ms per image on static input

YOLOX-Nano on Ubuntu

Real-time video inference with Python

Live object detection at 10–20 FPS

Uses OpenCV, GStreamer, and VART runner

Folder Structure 🗂️

.
├── vivado_project/         # Vivado project files
├── petalinux_resnet/       # ResNet-50 implementation
├── ubuntu_yolox/           # YOLOX Python pipeline
├── docs/                   # PDF report, screenshots, diagrams
└── README.md

Future Work 🔭

Combine classification and detection into one pipeline

Add YOLOv5 or segmentation models

Build a remote GUI or web interface

Expand to mobile robotics or surveillance use cases

