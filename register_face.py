import cv2
import os

# Ask for person's name
person_name = input("Enter person's name: ").strip()

# Create folder to save images
dataset_path = "dataset"
person_path = os.path.join(dataset_path, person_name)

os.makedirs(person_path, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0)

count = 0
max_images = 20

print("\nPress 'q' to quit.")
print(f"Capturing {max_images} face images...\n")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to access webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        count += 1

        filename = os.path.join(person_path, f"{count}.jpg")

        cv2.imwrite(filename, face)

        cv2.rectangle(frame,
                      (x, y),
                      (x+w, y+h),
                      (0, 255, 0),
                      2)

        cv2.putText(frame,
                    f"Captured: {count}/{max_images}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2)

        if count >= max_images:
            break

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nDone! Saved {count} images in '{person_path}'")