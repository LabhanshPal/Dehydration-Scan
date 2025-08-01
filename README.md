💧 DEHYDRATION SCAN

Dehydration Scan is a deep learning-based web application that detects dehydration in infants through facial images using a trained MobileNetV2 model. The application aims to assist in early-stage identification of dehydration symptoms for better child health monitoring.


🧠 PROJECT DESCRIPTION

This application utilizes computer vision to classify facial images of infants into two categories:

DEHYDRATED

HEALTHY (HYDRATED)

The front-end is built using Streamlit, providing an easy-to-use web interface for users to upload an image and receive instant predictions.


📁 PROJECT STRUCTURE
├── Dataset_Split/
│   ├── Train/
│   ├── Test/
│   └── Validation/
├── Preprocessed/
│   ├── Dehydrated_Infants/
│   └── Healthy_Infants/
├── dataset/
│   ├── Dehydrated_infants/
│   └── Healthy_infants/
├── final_project/
│   ├── app.py                  # Streamlit Web App
│   ├── dehydration.py          # Model Training and Preprocessing
│   ├── cv2_handler.py          # OpenCV Handler (Experimental)
├── dehydration_model.h5        # Trained Model File
├── requirements.txt            # Required Python Packages


🚀 DEPLOYMENT

The application has been successfully deployed and can be run locally or hosted on cloud platforms.


✅ HOW TO RUN LOCALLY

1. Clone the Repository:
git clone https://github.com/LabhanshPal/Dehydration-Scan.git
cd Dehydration-Scan/final_project

2. Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  (Windows: venv\Scripts\activate)

3. Install Required Dependencies:
pip install -r ../requirements.txt

4. Launch the App:
streamlit run app.py


⚙️ TECH STACK

Frontend / UI: Streamlit (Dark Theme)

Model: MobileNetV2 (TensorFlow / Keras)

Preprocessing: OpenCV, NumPy

Visualization (optional): Matplotlib


📸 FEATURES

Upload infant face images (JPG / PNG supported)

Instant prediction as "Dehydrated" or "Healthy"

Interactive, clean UI with dark mode and animation effects

Handles real-time webcam or file-based image predictions


🧠 Model Details

Base Model: MobileNetV2 (pre-trained on ImageNet)

Input Size: 224x224 pixels

Output: Binary Classification - Hydrated / Dehydrated

Fine-tuning: Last few layers fine-tuned on infant image dataset

Augmentation: Horizontal flip, brightness/zoom variations

Loss Function: Binary Crossentropy

Optimizer: Adam (learning rate = 0.0001)


📦 Dataset

The dataset used in this project contains two categories:

Healthy_Infants: Infants without visible signs of dehydration

Dehydrated_Infants: Infants with visible signs like sunken eyes, dry lips, etc.

Note: The dataset is not shared here due to privacy. Replace with your own dataset if needed.



⚠️ Edge Cases Tested

🟡 Blank images → Flagged as Hydrated (low confidence)

🟡 Cartoon/Non-human images → Mostly classified as Hydrated

✅ Real facial images → Correctly classified

🔜 Planned: Confidence score display and alert for "uncertain input"



✨ Future Improvements

Add support for video/frame-wise detection

Include additional dehydration symptoms through questionnaire

Improve robustness against irrelevant input

Deploy mobile app version



🙋‍♂️ AUTHOR

Labhansh Pal
B.Tech in Computer Science and Engineering (AI & DS)
GitHub: LabhanshPal


