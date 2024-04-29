import numpy as np
import cv2

def detect_features(frame, face_cascade, eye_cascade, background):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    mask = np.zeros_like(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)  # Fill face region with white
    
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            center = (int(x + ex + 0.5*ew), int(y + ey + 0.5*eh))
            radius = int(0.3 * (ew + eh))
            cv2.circle(mask, center, radius, (255, 255, 255), -1)  # Fill eye regions with white
    
    # Apply the mask to the original frame to get the foreground
    foreground = cv2.bitwise_and(frame, mask)

    # Invert the mask to get the background
    inverted_mask = cv2.bitwise_not(mask)
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))
    background_resized = cv2.bitwise_and(background_resized, inverted_mask)

    # Combine foreground and background
    result = cv2.add(foreground, background_resized)

    return result

def main():
    face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)

    background = cv2.imread('background.jpg')  # Load background image

    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = detect_features(frame, face_cascade, eye_cascade, background)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
