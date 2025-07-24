import matplotlib.pyplot as plt
import numpy as np

from src.data.ptbxl_loader import PTBXLDataset
from src.utils.preprocessing import ECGPreprocessor, ECGDataModule


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

    data_module = ECGDataModule(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        batch_size=32,
    )
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

if __name__ == "__main__":
    main()
