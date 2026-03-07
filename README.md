# Vision-Based Abnormal Behavior Detection System Using ROI and Deep Learning

This project is a computer vision system designed to detect abnormal behavior from surveillance cameras using Deep Learning and Region of Interest (ROI) analysis.

The system detects objects such as persons and vehicles from real-time video streams and analyzes their behavior based on how long they remain inside a predefined ROI. If an object stays in the monitored area longer than the configured threshold, the system records the event as abnormal and stores relevant evidence such as snapshots and video clips.

This project was developed as a senior project in the Computer Science program.

---

# System Architecture

The system consists of three main components:

1. **AI Detection Engine**
   - Built with Python
   - Uses YOLOv8 for object detection
   - Uses OpenCV for video processing
   - Performs ROI-based abnormal behavior detection

2. **API Server**
   - Built with Flask
   - Provides REST API endpoints
   - Streams live video frames
   - Serves event data to the dashboard

3. **Dashboard Application**
   - Built with Flutter
   - Displays live camera stream
   - Shows event timeline
   - Allows users to view recorded events

---

# Features

- Real-time object detection using YOLOv8
- ROI-based abnormal behavior detection
- Automatic event recording
- Snapshot capture
- Video clip recording
- Live video streaming
- Event timeline dashboard
- REST API integration

---

# Installation Guide

1. Clone the repository

```bash
git clone https://github.com/Ningnozy03/Vision-Based-Abnormal-Behavior-Detection-Using-ROI-and-Deep-Learning.git
cd Vision-Based-Abnormal-Behavior-Detection-Using-ROI-and-Deep-Learning

2. Create Python Virtual Environment
python -m venv .venv

.venv\Scripts\activate

3. Install Python Dependencies
pip install -r requirements.txt

Running the System

The system requires two main processes:

##Running the System AI Detection Engine  API Server
Step 1: Start AI Detection Engine
Run the detection system: python run_v2_with_clip.py

Step 2: Start API Server
Open another terminal and run: python api_server.py
The API server will start at:http://localhost:8000
Output Data

Detected events are stored in:
outputs/
Structure:
outputs/
 ├── events/
 │   └── snapshot images
 ├── clips/
 │   └── recorded video clips
 └── logs/
     └── events.csv
The CSV file contains event metadata including
timestamp

object class

duration inside ROI

snapshot path

clip path

In dashboard part I will write in other repository 
