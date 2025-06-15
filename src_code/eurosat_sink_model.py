import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ==== CONFIG ====
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
BASE_PATH = r"C:\Users\Pooja Kodgirwar\Downloads\GM1\models\archive (5)\EuroSAT"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
VAL_CSV = os.path.join(BASE_PATH, "validation.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")

label_encoder = LabelEncoder()

# ==== VALIDATION ====
def filter_valid_paths(df):
    full_paths = [os.path.join(BASE_PATH, f) for f in df["Filename"]]
    valid_rows = [os.path.exists(p) for p in full_paths]
    df_valid = df[valid_rows].reset_index(drop=True)
    return df_valid

# ==== DATASET LOADER ====
def load_dataset(csv_path, is_training=False):
    df = pd.read_csv(csv_path)
    df = filter_valid_paths(df)  # ✅ Remove missing files
    paths = [os.path.join(BASE_PATH, f) for f in df["Filename"]]
    labels = df["ClassName"]
    if is_training:
        labels = label_encoder.fit_transform(labels)
    else:
        labels = label_encoder.transform(labels)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _process(p, l):
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = preprocess_input(img)
        if is_training:
            img = tf.image.random_flip_left_right(img)
        return img, l

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    return ds, df

# ==== MODEL ====
def build_model(num_classes):
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model, base

# ==== MAIN ====
def main():
    train_ds, train_df = load_dataset(TRAIN_CSV, is_training=True)
    val_ds, val_df = load_dataset(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    test_df = filter_valid_paths(test_df)  # ✅

    model, base = build_model(num_classes=len(label_encoder.classes_))

    print("✅ Training base model...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    #print("🔁 Fine-tuning...")
    #base.trainable = True
    #model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save("eurosat_model.keras")
    print("✅ Model saved as eurosat_model.keras")

    print("🧪 Evaluating on test set...")
    X_test = []
    paths = [os.path.join(BASE_PATH, f) for f in test_df["Filename"]]
    for p in paths:
        img = image.load_img(p, target_size=(IMG_SIZE, IMG_SIZE))
        arr = image.img_to_array(img)
        arr = preprocess_input(arr)
        X_test.append(arr)
    X_test = np.array(X_test)
    y_test = label_encoder.transform(test_df["ClassName"])
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {acc:.4f}")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    main()
