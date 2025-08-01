import os
# import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Preprocessing function
def preprocess_images(input_folder, output_folder, img_size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    processed_count = 0

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        img = img / 255.0
        img = (img * 255).astype(np.uint8)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, img)
        processed_count += 1

    print(f"Processed {processed_count} images in {input_folder}")

# Preprocess both classes
if __name__ == "__main__":
    # Only runs when file is executed directly, not when imported in app
    preprocess_images("Dataset/Healthy_Infants", "Preprocessed/Healthy_Infants")
    preprocess_images("Dataset/Dehydrated_Infants", "Preprocessed/Dehydrated_Infants")

# Dataset splitting
INPUT_DIR = "Preprocessed"
OUTPUT_DIR = "Dataset_Split"

for subset in ["Train", "Validation", "Test"]:
    for category in ["Healthy_Infants", "Dehydrated_Infants"]:
        os.makedirs(os.path.join(OUTPUT_DIR, subset, category), exist_ok=True)

def split_and_copy(category, train_ratio=0.6, val_ratio=0.2):
    source_folder = os.path.join(INPUT_DIR, category)
    images = os.listdir(source_folder)

    train_imgs, test_imgs = train_test_split(images, test_size=1 - train_ratio)
    val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5)

    for img in train_imgs:
        shutil.copy(os.path.join(source_folder, img), os.path.join(OUTPUT_DIR, "Train", category))
    for img in val_imgs:
        shutil.copy(os.path.join(source_folder, img), os.path.join(OUTPUT_DIR, "Validation", category))
    for img in test_imgs:
        shutil.copy(os.path.join(source_folder, img), os.path.join(OUTPUT_DIR, "Test", category))

split_and_copy("Healthy_Infants")
split_and_copy("Dehydrated_Infants")

print("Dataset successfully split into Train, Validation, and Test sets!")

# Data generators (Augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
val_test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    "Dataset_Split/Train",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_data = val_test_datagen.flow_from_directory(
    "Dataset_Split/Validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

test_data = val_test_datagen.flow_from_directory(
    "Dataset_Split/Test",
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print("Data Augmentation Applied to Training Set Only!")


# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model Compiled Successfully!")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    steps_per_epoch=len(train_data),
    validation_steps=len(val_data),
    verbose=1
)

# Saving model
model.save("dehydration_model.h5")
print("Training Completed!")

# Printing loss and accuracy of model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

#Testing on sample image
sample_image, sample_label = next(iter(test_data))
sample_image = np.expand_dims(sample_image[0], axis=0)

prediction = model.predict(sample_image)
print(f"Predicted Value: {prediction}")
print("Dehydration Detected" if prediction[0][0] > 0.65 else "No Dehydration")

# Knowing the size of  augmented vs original image data
# batch_size = 32
# epochs = 20
# print(f"Total Training Images Before Augmentation: {len(train_data)}")
# print(f"Total Augmented Images Per Epoch: {len(train_data) * batch_size}")
# print(f"Total Augmented Images After {epochs} Epochs: {len(train_data) * batch_size * epochs}")

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Manual prediction
image_path1 = "Dataset_Split/Test/Healthy_Infants/Screenshot 2025-03-18 005800.png"
sample_image1 = cv2.imread(image_path1)
sample_image1 = cv2.resize(sample_image1, (224, 224))
sample_image1 = np.expand_dims(sample_image1, axis=0)

prediction1 = model.predict(sample_image1)
print(f"Predicted Value: {prediction1}")
print("Dehydration Detected" if prediction1[0][0] > 0.5 else "No Dehydration")

# Batch prediction testing
# healthy_folder = "Dataset_Split/Test/Healthy_Infants"
# dehydrated_folder = "Dataset_Split/Test/Dehydrated_Infants"
# def test_images(folder, label):
#     for filename in os.listdir(folder)[:20]:
#         img_path = os.path.join(folder, filename)
#         img = cv2.imread(img_path)
#         img = cv2.resize(img, (224, 224))
#         img = img.astype(np.float32) / 255.0
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.expand_dims(img, axis=0)

#         pred = model.predict(img)[0][0]

#         print("pred : dehydrated" if pred > 0.3 else "pred : healthy")

# test_images(healthy_folder, "Healthy")
# test_images(dehydrated_folder, "Dehydrated")
