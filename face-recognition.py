import cv2
import sys

def main():
    # Load the cascade classifier
    cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Check if the cascade file loaded correctly
    if face_cascade.empty():
        print(f"Error: Could not load cascade classifier from {cascade_path}")
        print("Ensure the file exists in the same directory.")
        sys.exit(1)

    # Initialize video capture from default camera (0)
    video_capture = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not video_capture.isOpened():
        print("Error: Could not open video source.")
        sys.exit(1)

    print("Face detection started. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to grayscale (Haar cascades work better on grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Exit the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy the window
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
