import cv2
import os
import time

# -----------------------------
# 1. Ask for person's ID
# -----------------------------

person_id = input("Enter person ID (example: person_001): ").strip()

if person_id == "":
    print("Error: Person ID cannot be empty.")
    exit()

# -----------------------------
# 2. Create person's folder
# -----------------------------

save_folder = os.path.join("dataset", "raw", person_id)

os.makedirs(save_folder, exist_ok=True)

print()
print("Saving images to:", save_folder)

# -----------------------------
# 3. Open webcam
# -----------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# -----------------------------
# 4. Load face detector
# -----------------------------

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# 5. Variables
# -----------------------------

image_count = 0
max_images = 20

print()
print("Camera started.")
print("Look at the camera.")
print("Move your face slightly in different directions.")
print("Press Q to stop.")
print()

# Small delay before starting
time.sleep(2)

# -----------------------------
# 6. Start collecting faces
# -----------------------------

while image_count < max_images:

    success, frame = camera.read()

    if not success:
        print("Error: Could not read camera.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # Process detected faces
    for (x, y, w, h) in faces:

        # Crop the face
        face = frame[y:y+h, x:x+w]

        # Resize face
        face = cv2.resize(face, (160, 160))

        # Create filename
        filename = os.path.join(
            save_folder,
            f"face_{image_count + 1:03d}.jpg"
        )

        # Save image
        cv2.imwrite(filename, face)

        image_count += 1

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        break

    # Show progress
    cv2.putText(
        frame,
        f"Images: {image_count}/{max_images}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show camera
    cv2.imshow("Face Dataset Collection", frame)

    # Press Q to stop
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

# -----------------------------
# 7. Close everything
# -----------------------------

camera.release()
cv2.destroyAllWindows()

print()
print("Face collection completed.")
print("Total images:", image_count)
print("Saved in:", save_folder)