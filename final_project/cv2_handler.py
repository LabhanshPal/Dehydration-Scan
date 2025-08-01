import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the Keras 3-compatible model
model = load_model("dehydration_model.h5")

def capture_and_predict():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        return

    print("Press 's' to take a snapshot for dehydration prediction.")
    print("Press 'q' to quit the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam - Press 's' to capture, 'q' to quit", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            # Preprocess
            img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            pred = model.predict(img)[0][0]
            print(f"Prediction Score: {pred}")
            print("Dehydration Detected" if pred > 0.65 else "No Dehydration")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run
capture_and_predict()
