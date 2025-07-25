import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from tqdm import tqdm
import numpy as np
from src.data.ptbxl_loader import PTBXLDataset
from src.utils.preprocessing import ECGPreprocessor
from src.models.cnn1d import create_cnn1d_model
from src.utils.evaluation import save_confusion_matrix, save_training_history


def main():
    data_path = "dataset"
    ptbxl_dataset = PTBXLDataset(data_path)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = (
        ptbxl_dataset.get_dataset()
    )

    print(f"Dataset loaded")
    print(f"Train set shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
    print(f"Test set shape: {X_test.shape}, {y_test.shape}")
    print(f"Class names: {class_names}")

    preprocessor = ECGPreprocessor(sampling_rate=100, target_length=1000)
    X_train = preprocessor.preprocess(X_train, apply_filters=True)
    X_val = preprocessor.preprocess(X_val, apply_filters=True)
    X_test = preprocessor.preprocess(X_test, apply_filters=True)

    y_train = to_categorical(y_train, num_classes=len(class_names))
    y_val = to_categorical(y_val, num_classes=len(class_names))
    y_test = to_categorical(y_test, num_classes=len(class_names))

    model = create_cnn1d_model(
        input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=len(class_names)
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    print("TensorFlow devices:", tf.config.list_physical_devices())

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
    )

    test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=32, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    y_pred = np.argmax(model.predict(X_test, batch_size=32), axis=1)
    y_true = np.argmax(y_test, axis=1)
    save_confusion_matrix(y_true, y_pred, class_names, output_path="confusion_matrix.png")
    save_training_history(history, output_path="training_history.png")


if __name__ == "__main__":
    main()

