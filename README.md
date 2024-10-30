Object Detection and Distance Estimation with DepthAI and YOLOv6

This repository contains the code for my final university project, which was pivotal in earning my bachelor's degree. The project demonstrates a Python-based system for real-time object detection and distance estimation using Luxonis' DepthAI platform with a stereo camera setup. The YOLOv6 model is utilized for accurate object detection, and DepthAI's API enables precise depth calculations.
Project Overview

This project integrates:

    * Luxonis DepthAI Stereo Camera: Captures stereo images, which are processed to estimate distances to detected objects.
    * YOLOv6 Object Detection: Uses a pre-trained YOLOv6 model to detect a wide range of objects.
    * Depth Calculation: Employs stereo vision principles to estimate object distance using disparity maps.

Key Features

    Real-Time Object Detection: Detects over 80 classes, including vehicles, animals, and everyday objects, with YOLOv6.
    Depth Estimation: Calculates distance to detected objects based on the stereo camera’s depth information.
    Infrared Support: Utilizes IR projection for improved depth perception in low-light environments.

Installation

To install the necessary dependencies, run:

bash

pip install -r requirements.txt

Requirements:

    Python 3.8+
    opencv-python==4.5.5.64
    depthai==2.26.0.0
    numpy

Usage

    Connect the DepthAI device.
    Run the main script:

    bash

    python main.py

    The system will display live object detection with distance estimates overlayed on each detected object.

Future Improvements

    Enhanced object tracking.
    Optimized distance calculations for varying environmental conditions.
