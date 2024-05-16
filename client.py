import numpy as np
import cv2

def fetch_stream(ip, port, camera_id):
    stream_url = f'http://{ip}:{port}/video_feed/{camera_id}'
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Failed to open stream from {ip}:{port}/{camera_id}")
    return cap

def detect_all_features(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
   
    for (x, y, w, h) in faces:
        # Draw blue rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return frame

def main():
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
   
    if face_cascade.empty():
        print("Failed to load Haar cascades.")
        return
   
    # List of IP addresses, port numbers, and camera IDs of each camera server
    camera_servers = [
        {'ip': '192.168.1.22', 'port': 5000, 'cameras': ['camera0', 'camera1','camera2']},
       
        # Add more camera servers as needed
    ]
   
    caps = []
    for server in camera_servers:
        for camera_id in server['cameras']:
            cap = fetch_stream(server['ip'], server['port'], camera_id)
            caps.append(cap)

    # Get screen dimensions
    screen_width = 1366  # Update with actual screen width if needed
    screen_height = 768  # Update with actual screen height if needed
   
    # Determine size for each frame to fit within screen width and height
    num_columns = 3  # Number of columns you want
    max_frame_height = screen_height // 2  # Max frame height is half the screen height
    frame_width = screen_width // num_columns
    frame_height = frame_width * 3 // 4  # Assuming a 4:3 aspect ratio

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frame = detect_all_features(frame, face_cascade)
                frame = cv2.resize(frame, (frame_width, frame_height))
                frames.append(frame)
            else:
                frames.append(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))  # Placeholder for failed streams

        if frames:
            # Calculate number of rows needed
            num_rows = (len(frames) + num_columns - 1) // num_columns

            # Ensure the total height doesn't exceed screen height
            total_height = num_rows * frame_height
            if total_height > screen_height:
                frame_height = screen_height // num_rows
                frame_width = frame_height * 4 // 3  # Maintain aspect ratio
                frames = [cv2.resize(frame, (frame_width, frame_height)) for frame in frames]

            combined_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            current_y = 0
            current_x = 0
            for frame in frames:
                if current_x + frame_width > screen_width:
                    current_x = 0
                    current_y += frame_height
                combined_frame[current_y:current_y + frame_height, current_x:current_x + frame_width] = frame
                current_x += frame_width

            cv2.imshow('Facial Detection', combined_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()