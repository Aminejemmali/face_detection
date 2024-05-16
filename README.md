# Face Detection System

This project implements a simple face and eye detection system using OpenCV in Python. It uses Haar Cascades to detect faces and eyes in real-time through a webcam feed.

## Installation (phase 1 and 2)

Install the required Python packages:

```bash
pip install opencv-python numpy
```

## Usage (phase 1 and 2)

To run the face detection system, execute the following command in the root directory of the project:

```bash
python detection.py
```

Ensure your webcam is enabled and properly configured as the application uses the default camera device.

## Phase 1
The project has successfully passed phase 1, including the detection of objects in one camera device.
## Phase 2: Dual Camera Support
Second phase completed, we plan to enhance the system to support simultaneous detection using three diffrent camera devices (integrated,external,network). This will allow for more robust face detection in diverse scenarios.
## Phase 3: Gathering and displaying LAN Streams on a central Machine
In this phase,each machine should run a script to stream its camera feed using a simple HTTP flask server .Finally ,The central machine will run a script to collect these streams and display them in a single window

## Installation

Camera server machines
```bash
pip install opencv-python flask
```

Central Machine
```bash
pip install opencv-python numpy
```

## Usage 

Streaming Script(camera_server.py)

```bash
python camera_server.py
```

On the central machine (client.py)

```bash
python client.py
```