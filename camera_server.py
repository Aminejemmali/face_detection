# multi_camera_server.py
import cv2
from flask import Flask, Response, request

app = Flask(__name__)
stream_url = 'http://192.168.1.15:4747/video'
# Dictionary to hold video capture objects for each camera
cameras = {
    'camera0': cv2.VideoCapture(0),
    'camera1': cv2.VideoCapture(1),
    'camera2': cv2.VideoCapture(stream_url)  # Add more cameras as needed
}

def generate_frames(camera_id):
    cap = cameras[camera_id]
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in cameras:
        return "Camera not found", 404
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
