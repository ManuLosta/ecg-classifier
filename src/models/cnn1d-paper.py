import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Tuple


class LargeBlock(nn.Module):
    """Parallel convolutional block with multiple kernel sizes."""

    def __init__(self, input_channels: int, output_channels: int = 64):
        super(LargeBlock, self).__init__()

        # Branch 1: kernel size 7
        self.branch1_conv = nn.Conv1d(
            input_channels, output_channels, kernel_size=7, padding=3)
        self.branch1_bn = nn.BatchNorm1d(output_channels)

        # Branch 2: kernel size 5
        self.branch2_conv = nn.Conv1d(
            input_channels, output_channels, kernel_size=5, padding=2)
        self.branch2_bn = nn.BatchNorm1d(output_channels)

        # Branch 3: kernel size 3
        self.branch3_conv = nn.Conv1d(
            input_channels, output_channels, kernel_size=3, padding=1)
        self.branch3_bn = nn.BatchNorm1d(output_channels)

        # Branch 4: kernel size 1
        self.branch4_conv = nn.Conv1d(
            input_channels, output_channels, kernel_size=1, padding=0)
        self.branch4_bn = nn.BatchNorm1d(output_channels)

    def forward(self, x):
        # Branch 1
        branch1 = F.relu(self.branch1_bn(self.branch1_conv(x)))

        # Branch 2
        branch2 = F.relu(self.branch2_bn(self.branch2_conv(x)))

        # Branch 3
        branch3 = F.relu(self.branch3_bn(self.branch3_conv(x)))

        # Branch 4
        branch4 = F.relu(self.branch4_bn(self.branch4_conv(x)))

        # Concatenate all branches
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)


class SmallBlock(nn.Module):
    """Small convolutional block with dropout and pooling."""

    def __init__(self, input_channels: int, output_channels: int = 64, kernel_size: int = 7, dropout_rate: float = 0.3):
        super(SmallBlock, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv1d(input_channels, output_channels,
                              kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(output_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        x = self.pool(x)
        return x


class CNN1DPaper(nn.Module):
    """CNN1D model based on the paper architecture with parallel branches."""

    def __init__(self, input_channels: int, num_classes: int):
        super(CNN1DPaper, self).__init__()

        # Large block with parallel branches (outputs 64*4 = 256 channels)
        self.large_block = LargeBlock(input_channels, output_channels=64)

        # Small blocks taking the concatenated output (256 channels)
        self.small_block1 = SmallBlock(
            256, output_channels=64, kernel_size=7, dropout_rate=0.3)
        self.small_block2 = SmallBlock(
            256, output_channels=64, kernel_size=5, dropout_rate=0.3)
        self.small_block3 = SmallBlock(
            256, output_channels=64, kernel_size=1, dropout_rate=0.3)

        # Global average pooling and classification
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        # 64*3 = 192 from concatenated small blocks
        self.fc1 = nn.Linear(192, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout_fc = nn.Dropout(0.3)

    def forward(self, x):
        # Large block with parallel branches
        x = self.large_block(x)

        # Small blocks (parallel processing of the merged features)
        sb1 = self.small_block1(x)
        sb2 = self.small_block2(x)
        sb3 = self.small_block3(x)

        # Concatenate small block outputs
        x = torch.cat([sb1, sb2, sb3], dim=1)

        # Global average pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        # Classification layers
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x


def create_cnn1d_model(input_shape: Tuple[int, int], num_classes: int) -> CNN1DPaper:
    """
    Create a CNN1D model with the paper architecture.

    Args:
        input_shape: Tuple of (sequence_length, input_channels)
        num_classes: Number of output classes

    Returns:
        CNN1DPaper model
    """
    # In PyTorch Conv1d, input_channels is the second dimension
    input_channels = input_shape[1] if len(
        input_shape) == 2 else input_shape[0]
    return CNN1DPaper(input_channels=input_channels, num_classes=num_classes)


def save_model(model: nn.Module, filepath: str, model_state_dict_only: bool = False):
    """
    Save the model to the specified filepath.

    Args:
        model: PyTorch model to save
        filepath: Path where to save the model
        model_state_dict_only: If True, save only state_dict, otherwise save entire model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if model_state_dict_only:
        torch.save(model.state_dict(), filepath)
        print(f"Model state dict saved to {filepath}")
    else:
        torch.save(model, filepath)
        print(f"Complete model saved to {filepath}")


def save_model_to_trained_dir(model: nn.Module, model_name: str, models_dir: str = "models/trained"):
    """
    Save model to the trained models directory with .keras extension.

    Args:
        model: PyTorch model to save
        model_name: Name for the model file (without extension)
        models_dir: Directory where to save the model
    """
    if not model_name.endswith('.keras'):
        model_name += '.keras'

    filepath = os.path.join(models_dir, model_name)
    save_model(model, filepath, model_state_dict_only=True)
    return filepath


def load_model(filepath: str, model_class=None, **model_kwargs):
    """
    Load a saved model.

    Args:
        filepath: Path to the saved model
        model_class: Model class to instantiate (required if loading state_dict)
        **model_kwargs: Arguments to pass to model constructor

    Returns:
        Loaded PyTorch model
    """
    if filepath.endswith('.pth') and model_class is not None:
        # Load state dict
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
        return model
    else:
        # Load complete model
        return torch.load(filepath, map_location='cpu')


# Example usage:
if __name__ == "__main__":
    # Example model creation
    # (sequence_length, channels) for ECG with 12 leads
    input_shape = (1000, 12)
    num_classes = 5

    model = create_cnn1d_model(input_shape, num_classes)
    print(f"Model created with input shape {
          input_shape} and {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Example forward pass
    batch_size = 8
    # (batch, channels, sequence_length)
    x = torch.randn(batch_size, input_shape[1], input_shape[0])
    output = model(x)
    print(f"Output shape: {output.shape}")

    # Example save
    save_model_to_trained_dir(model, "cnn1d_paper_example")
