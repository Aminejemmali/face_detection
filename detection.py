import numpy as np
import cv2

def detect_features(frame, face_cascade, eye_cascade, background):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    mask = np.zeros_like(frame)
    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)  # Fill face region with white
    
        roi_gray = gray[y:y+h, x:x+w]
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

    ret, frozen_frame = cap.read()  # Capture frame to freeze
    if not ret:
        print("Failed to capture frame")
        return

    frozen_frame = detect_features(frozen_frame, face_cascade, eye_cascade, background)
    frozen_frame_text = frozen_frame.copy()

    # Add text overlay to frozen frame
    text = 'Will start in the next update'
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    text_x = (frozen_frame.shape[1] - text_size[0]) // 2
    text_y = (frozen_frame.shape[0] + text_size[1]) // 2
    cv2.putText(frozen_frame_text, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    while(True):
        ret, live_frame = cap.read()
        if not ret:
            break

        live_frame = detect_features(live_frame, face_cascade, eye_cascade, background)

        # Combine frozen and live frames horizontally
        combined_frame = np.hstack((frozen_frame_text, live_frame))

        cv2.imshow('Frozen and Live Frame', combined_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
