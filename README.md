Title: Drone Surveillance ATR (Automatic Target Recognition) System
Overview: An end-to-end computer vision pipeline that detects and tracks vehicles and pedestrians from a drone's perspective.
Tech Stack: Python, YOLOv8 (Ultralytics), OpenCV, Tkinter, Google Colab (for GPU training).
Engineering Highlights:

Fine-tuned a YOLOv8 Nano model on the VisDrone dataset.

Implemented object tracking (BoT-SORT) to eliminate bounding-box flickering in 30fps video.

Engineered dynamic resolution scaling so the tactical UI perfectly fits both 4K images and low-res video.

Packaged the raw script into a local desktop application using Tkinter.
