import cv2

# Open the webcam
camera = cv2.VideoCapture(0)

# Check whether the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started.")
print("Press Q to quit.")

# Load OpenCV face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:

    # Capture one frame from camera
    success, frame = camera.read()

    if not success:
        print("Error: Could not read frame.")
        break

    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # Draw rectangle around detected faces
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Show number of faces
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display camera
    cv2.imshow("Face Detection Test", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close camera
camera.release()
cv2.destroyAllWindows()