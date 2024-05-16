import numpy as np
import cv2

def detect_all_features(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        # Draw blue rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Display an alert for detected face
        cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    return frame

def main():
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    
    if face_cascade.empty():
        print("Failed to load Haar cascades.")
        return

    cap_primary = cv2.VideoCapture(0)  # Internal webcam
    cap_secondary = cv2.VideoCapture(1)  # External webcam
    stream_url = 'ip_from_droid_cam'
    cap_third = cv2.VideoCapture(stream_url)
    if not cap_primary.isOpened() or not cap_secondary.isOpened() or not cap_third.isOpened():
        print("Failed to open webcams.")
        return
    
    while True:
        ret1, frame1 = cap_primary.read()  # Capture frame from internal webcam
        ret2, frame2 = cap_secondary.read()  # Capture frame from external webcam
        ret3, frame3 = cap_third.read()
        if not ret1 or not ret2 or not ret3:
            break

        frame1 = detect_all_features(frame1, face_cascade)
        frame2 = detect_all_features(frame2, face_cascade)
        frame3 = detect_all_features(frame3, face_cascade)

        # Add titles for camera feeds
        cv2.putText(frame1, 'Integrated Camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame2, 'External Camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame3, 'Network Camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        # Resize frames to have same height
        h1, w1 = frame1.shape[:2] 
        h2, w2 = frame2.shape[:2]
        h3 , w3 = frame3.shape[:2]
        max_height = max(h1, h2,h3)
        frame1_resized = cv2.resize(frame1, (int(w1 * max_height / h1), max_height))
        frame2_resized = cv2.resize(frame2, (int(w2 * max_height / h2), max_height))
        frame3_resized = cv2.resize(frame3, (int(w3 * max_height / h3), max_height))

        # Combine frames horizontally
        combined_frame = np.hstack((frame1_resized, frame2_resized,frame3_resized))

        cv2.imshow('Facial Detection', combined_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # Release video capture resources
    cap_primary.release()
    cap_secondary.release()
    # Close OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
