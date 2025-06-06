import cv2
import os

# Path to Haar Cascade file
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Input and output directories
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'output_images'

# Load the Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Make sure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        img_path = os.path.join(INPUT_DIR, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Failed to load image: {filename}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        print(f"{filename}: {len(faces)} face(s) detected.")

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the output image
        output_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(output_path, image)

        # Optionally, show the image (press any key to continue)
        # cv2.imshow('Detected Faces', image)
        # cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
