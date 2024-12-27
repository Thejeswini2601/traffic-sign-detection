import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Path to the dataset folder
data_dir = 'Indian-Traffic Sign-Dataset\Images'

# Initialize data and labels
data = []
labels = []

# Image size for resizing
IMG_SIZE = 64

# Load images and labels
for class_id in range(59):  # Assuming classes are numbered 0 to 58
    folder_path = os.path.join(data_dir, str(class_id))
    if not os.path.exists(folder_path):
        continue

    for img_file in os.listdir(folder_path):
        try:
            # Load image
            img_path = os.path.join(folder_path, img_file)
            img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
            img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize image to [0, 1]

            # Append to data and labels
            data.append(img)
            labels.append(class_id)
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")

data = np.array(data, dtype="float32")
labels = np.array(labels, dtype="int")

labels = to_categorical(labels, num_classes=59)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(59, activation='softmax')  # 59 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=[early_stopping]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

'''ACCURACY AND OUTPUT
Epoch 1/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 38s 105ms/step - accuracy: 0.1228 - loss: 3.5949 - val_accuracy: 0.4877 - val_loss: 1.9966
Epoch 2/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 41s 104ms/step - accuracy: 0.3939 - loss: 2.1516 - val_accuracy: 0.6615 - val_loss: 1.2681
Epoch 3/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 37s 105ms/step - accuracy: 0.5374 - loss: 1.5645 - val_accuracy: 0.7030 - val_loss: 1.0336
Epoch 4/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 36s 104ms/step - accuracy: 0.5961 - loss: 1.2807 - val_accuracy: 0.7331 - val_loss: 0.8880
Epoch 5/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 41s 104ms/step - accuracy: 0.6359 - loss: 1.1070 - val_accuracy: 0.7574 - val_loss: 0.7983
Epoch 6/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 36s 103ms/step - accuracy: 0.6638 - loss: 1.0075 - val_accuracy: 0.7671 - val_loss: 0.7588
Epoch 7/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 41s 105ms/step - accuracy: 0.6998 - loss: 0.9088 - val_accuracy: 0.7789 - val_loss: 0.7073
Epoch 8/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 36s 104ms/step - accuracy: 0.7192 - loss: 0.8466 - val_accuracy: 0.7832 - val_loss: 0.6754
Epoch 9/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 36s 103ms/step - accuracy: 0.7382 - loss: 0.7765 - val_accuracy: 0.7807 - val_loss: 0.6835
Epoch 10/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 41s 104ms/step - accuracy: 0.7590 - loss: 0.7274 - val_accuracy: 0.7853 - val_loss: 0.6889
Epoch 11/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 37s 105ms/step - accuracy: 0.7638 - loss: 0.6991 - val_accuracy: 0.7875 - val_loss: 0.6481
Epoch 12/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 36s 104ms/step - accuracy: 0.7792 - loss: 0.6407 - val_accuracy: 0.8029 - val_loss: 0.6245
Epoch 13/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 37s 105ms/step - accuracy: 0.7875 - loss: 0.6044 - val_accuracy: 0.7971 - val_loss: 0.6469
Epoch 14/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 41s 104ms/step - accuracy: 0.8017 - loss: 0.5846 - val_accuracy: 0.7986 - val_loss: 0.6469
Epoch 15/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 41s 104ms/step - accuracy: 0.7974 - loss: 0.5837 - val_accuracy: 0.8029 - val_loss: 0.6267
Epoch 16/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 36s 104ms/step - accuracy: 0.8096 - loss: 0.5536 - val_accuracy: 0.8007 - val_loss: 0.6462
Epoch 17/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 37s 105ms/step - accuracy: 0.8133 - loss: 0.5334 - val_accuracy: 0.8014 - val_loss: 0.6239
Epoch 18/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 37s 106ms/step - accuracy: 0.8314 - loss: 0.5091 - val_accuracy: 0.8032 - val_loss: 0.6394
Epoch 19/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 37s 106ms/step - accuracy: 0.8215 - loss: 0.5142 - val_accuracy: 0.8107 - val_loss: 0.6195
Epoch 20/20
350/350 ━━━━━━━━━━━━━━━━━━━━ 36s 104ms/step - accuracy: 0.8361 - loss: 0.4961 - val_accuracy: 0.8107 - val_loss: 0.6440
88/88 ━━━━━━━━━━━━━━━━━━━━ 2s 26ms/step - accuracy: 0.8189 - loss: 0.5729
Test Accuracy: 0.81'''

# Save the trained model
model.save("traffic_sign_classifier.h5")
print("Model saved as 'traffic_sign_classifier.h5'")

import cv2
from matplotlib import pyplot as plt

# Function to make a prediction and display the image
def predict_and_display(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) 

    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Class names
    class_names = [
        "Give way", "No entry", "One-way traffic", "One-way traffic", "No vehicles in both directions",
        "No entry for cycles", "No entry for goods vehicles", "No entry for pedestrians",
        "No entry for bullock carts", "No entry for hand carts", "No entry for motor vehicles",
        "Height limit", "Weight limit", "Axle weight limit", "Length limit", "No left turn",
        "No right turn", "No overtaking", "Maximum speed limit (90 km/h)", "Maximum speed limit (110 km/h)",
        "Horn prohibited", "No parking", "No stopping", "Turn left", "Turn right", "Steep descent",
        "Steep ascent", "Narrow road", "Narrow bridge", "Unprotected quay", "Road hump", "Dip",
        "Loose gravel", "Falling rocks", "Cattle", "Crossroads", "Side road junction",
        "Side road junction", "Oblique side road junction", "Oblique side road junction",
        "T-junction", "Y-junction", "Staggered side road junction", "Staggered side road junction",
        "Roundabout", "Guarded level crossing ahead", "Unguarded level crossing ahead",
        "Level crossing countdown marker", "Level crossing countdown marker", "Level crossing countdown marker",
        "Level crossing countdown marker", "Parking", "Bus stop", "First aid post", "Telephone",
        "Filling station", "Hotel", "Restaurant", "Refreshments"
    ]

    # Predicted class name
    predicted_label = class_names[predicted_class]

    height, width, _ = img.shape
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 255, 0), 5)  # Draw rectangle
    cv2.putText(img, predicted_label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Detected: {predicted_label}")
    plt.show()

# Example usage
predict_and_display("path-to-sample-input")