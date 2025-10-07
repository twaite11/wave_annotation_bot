import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
import re
from typing import List, Tuple

DATA_DIR = 'wave_dataset'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 25
EPOCHS_FINE_TUNE = 0
TEST_SIZE = 0.2
RANDOM_SEED = 42

def load_data(data_dir: str, img_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    filepaths = []
    labels = []

    score_pattern = re.compile(r'wave_score_(\d{2})_id_\d{3}\.png$')

    print(f"Scanning directory: {data_dir}...")

    if not os.path.exists(data_dir):
        print(f"Error: Directory '{data_dir}' not found. Please create it and add your images.")
        print("Simulating dummy data loading for demonstration purposes.")
        return np.random.rand(200, img_size[0], img_size[1], 3), np.random.randint(1, 11, 200), ["dummy"] * 200


    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            match = score_pattern.search(filename)
            if match:
                score = int(match.group(1))
                filepaths.append(os.path.join(data_dir, filename))
                labels.append(score)

    if not filepaths:
        print(f"Warning: Found 0 images matching the naming convention in '{data_dir}'.")
        print("Please ensure files are named like 'wave_score_05_id_123.png'.")
        return np.random.rand(200, img_size[0], img_size[1], 3), np.random.randint(1, 11, 200), ["dummy"] * 200

    print(f"Found {len(filepaths)} images. Loading and processing...")

    images = []
    for filepath in filepaths:
        img = load_img(filepath, target_size=img_size)
        img_array = img_to_array(img)
        images.append(img_array)

    X = np.array(images)
    Y = np.array(labels)

    return X, Y, filepaths

def build_regression_model(img_size: Tuple[int, int]) -> Tuple[Model, Model]:
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=img_size + (3,)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)

    predictions = Dense(1, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model

def train_model(X: np.ndarray, Y: np.ndarray):
    X_train, X_test, Y_train_raw, Y_test_raw = train_test_split(
        X, Y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    scaler = MinMaxScaler()
    scaler.fit(Y.reshape(-1, 1))
    Y_train = scaler.transform(Y_train_raw.reshape(-1, 1))
    Y_test = scaler.transform(Y_test_raw.reshape(-1, 1))

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    train_generator = train_datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE)
    validation_generator = test_datagen.flow(X_test, Y_test, batch_size=BATCH_SIZE)

    model, base_model = build_regression_model(IMG_SIZE)

    print("\n--- Training Model: Frozen VGG16 with Regularized Head ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    history_frozen = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=len(X_test) // BATCH_SIZE,
        epochs=EPOCHS_FROZEN,
        verbose=1
    )
    print("--- Training Complete ---")

    loss, mae = model.evaluate(test_datagen.flow(X_test, Y_test, batch_size=BATCH_SIZE, shuffle=False), verbose=0)

    raw_mae = mae * (scaler.data_max_[0] - scaler.data_min_[0])

    print(f"\nFinal Model Evaluation (Test Set):")
    print(f"Scaled Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Estimated Real Score Error (MAE): {raw_mae:.2f} points (e.g., predicted 7.5 instead of 7.0)")

    model.save('wave_quality_model.h5')
    print("\nModel saved successfully as 'wave_quality_model.h5'")


if __name__ == '__main__':
    X, Y, filepaths = load_data(DATA_DIR, IMG_SIZE)

    if len(X) > 0:
        train_model(X, Y)
    else:
        print("\nCould not proceed with training due to missing or incorrectly named images.")
