import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import List, Tuple


class LeadProcessor(nn.Module):
    """Process individual ECG lead with Conv2D + BatchNorm + ReLU + MaxPool."""
    
    def __init__(self, input_channels: int = 1):
        super(LeadProcessor, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=(7, 7), padding=3)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=3)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 33x30 → ~11x10
        return x


class TerritoryExpert(nn.Module):
    """Expert for specific ECG territory (Inferior, Lateral, Septal/Anterior, Global)."""
    
    def __init__(self, num_leads: int, expert_name: str = ""):
        super(TerritoryExpert, self).__init__()
        
        self.expert_name = expert_name
        self.num_leads = num_leads
        input_channels = 32 * num_leads  # 32 channels per lead
        
        self.conv = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout2d(0.25)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)  # ~11x10 → ~5x5
        x = self.dropout(x)
        return x


class InferiorExpert(TerritoryExpert):
    """Expert for Inferior territory: II, III, aVF (leads 1, 2, 5)."""
    
    def __init__(self):
        super(InferiorExpert, self).__init__(num_leads=3, expert_name="Inferior")


class LateralExpert(TerritoryExpert):
    """Expert for Lateral territory: I, aVL, V5, V6 (leads 0, 4, 10, 11)."""
    
    def __init__(self):
        super(LateralExpert, self).__init__(num_leads=4, expert_name="Lateral")


class SeptalAnteriorExpert(TerritoryExpert):
    """Expert for Septal/Anterior territory: V1, V2, V3, V4 (leads 6, 7, 8, 9)."""
    
    def __init__(self):
        super(SeptalAnteriorExpert, self).__init__(num_leads=4, expert_name="SeptalAnterior")


class GlobalExpert(TerritoryExpert):
    """Expert for Global view: all 12 leads for complex patterns."""
    
    def __init__(self):
        super(GlobalExpert, self).__init__(num_leads=12, expert_name="Global")


class ECGTerritoryMoE(nn.Module):
    """
    Mixture of Experts model for 12-lead ECG classification using territory-based experts.
    Each expert specializes in specific ECG territories for better diagnosis.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super(ECGTerritoryMoE, self).__init__()
        
        self.input_shape = input_shape  # (height, width, channels)
        self.num_classes = num_classes
        
        # Lead mapping for different territories
        # Standard 12-lead ECG order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
        self.lead_territories = {
            'inferior': [1, 2, 5],      # II, III, aVF (indices 1, 2, 5)
            'lateral': [0, 4, 10, 11],  # I, aVL, V5, V6 (indices 0, 4, 10, 11)
            'septal_anterior': [6, 7, 8, 9],  # V1, V2, V3, V4 (indices 6, 7, 8, 9)
            'global': list(range(12))   # All 12 leads
        }
        
        # 12 lead processors (one for each ECG lead)
        self.lead_processors = nn.ModuleList([
            LeadProcessor(input_channels=input_shape[2]) for _ in range(12)
        ])
        
        # Territory-based experts
        self.inferior_expert = InferiorExpert()
        self.lateral_expert = LateralExpert()
        self.septal_anterior_expert = SeptalAnteriorExpert()
        self.global_expert = GlobalExpert()
        
        # Calculate total output channels from all experts
        # Inferior: 64, Lateral: 64, Septal/Anterior: 64, Global: 64
        total_expert_channels = 64 * 4  # 256
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc1 = nn.Linear(total_expert_channels, 128)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, inputs: List[torch.Tensor]):
        """
        Forward pass with list of 12 lead inputs.
        
        Args:
            inputs: List of 12 tensors, each of shape (batch_size, channels, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        if len(inputs) != 12:
            raise ValueError(f"Expected 12 lead inputs, got {len(inputs)}")
        
        # Process each lead individually
        lead_outputs = []
        for i, lead_input in enumerate(inputs):
            processed = self.lead_processors[i](lead_input)
            lead_outputs.append(processed)
        
        # Territory-based expert processing
        expert_outputs = []
        
        # Inferior Expert: II, III, aVF
        inferior_leads = [lead_outputs[i] for i in self.lead_territories['inferior']]
        inferior_concat = torch.cat(inferior_leads, dim=1)
        inferior_output = self.inferior_expert(inferior_concat)
        expert_outputs.append(inferior_output)
        
        # Lateral Expert: I, aVL, V5, V6
        lateral_leads = [lead_outputs[i] for i in self.lead_territories['lateral']]
        lateral_concat = torch.cat(lateral_leads, dim=1)
        lateral_output = self.lateral_expert(lateral_concat)
        expert_outputs.append(lateral_output)
        
        # Septal/Anterior Expert: V1, V2, V3, V4
        septal_anterior_leads = [lead_outputs[i] for i in self.lead_territories['septal_anterior']]
        septal_anterior_concat = torch.cat(septal_anterior_leads, dim=1)
        septal_anterior_output = self.septal_anterior_expert(septal_anterior_concat)
        expert_outputs.append(septal_anterior_output)
        
        # Global Expert: All 12 leads
        global_concat = torch.cat(lead_outputs, dim=1)
        global_output = self.global_expert(global_concat)
        expert_outputs.append(global_output)
        
        # Merge all expert outputs
        merged = torch.cat(expert_outputs, dim=1)
        
        # Global average pooling
        x = self.global_avg_pool(merged)
        x = torch.flatten(x, 1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class ECGTerritoryMoESingleInput(nn.Module):
    """
    Territory-based MoE model that takes a single tensor input and splits it into leads.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super(ECGTerritoryMoESingleInput, self).__init__()
        
        self.territory_moe_model = ECGTerritoryMoE(input_shape, num_classes)
    
    def forward(self, x):
        """
        Forward pass with single input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, 12, height, width) or (batch_size, height, width, 12)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Split the input into 12 leads
        if x.shape[1] == 12:  # (batch, 12, height, width)
            lead_inputs = [x[:, i:i+1, :, :] for i in range(12)]
        elif x.shape[-1] == 12:  # (batch, height, width, 12)
            # Convert to (batch, 12, height, width) format
            x = x.permute(0, 3, 1, 2)
            lead_inputs = [x[:, i:i+1, :, :] for i in range(12)]
        else:
            raise ValueError(f"Input shape {x.shape} not supported. Expected last dim or second dim to be 12.")
        
        return self.territory_moe_model(lead_inputs)


class OpinionGroup(nn.Module):
    """Opinion group for processing multiple ECG leads together."""
    
    def __init__(self, input_channels: int):
        super(OpinionGroup, self).__init__()
        
        self.conv = nn.Conv2d(input_channels, 64, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.dropout = nn.Dropout2d(0.25)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x


class MultibranchCNN2D(nn.Module):
    """
    Multi-branch CNN for 12-lead ECG classification.
    Each lead is processed individually, then grouped into 4 opinion groups.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super(MultibranchCNN2D, self).__init__()
        
        self.input_shape = input_shape  # (height, width, channels)
        self.num_classes = num_classes
        
        # 12 lead processors (one for each ECG lead)
        self.lead_processors = nn.ModuleList([
            LeadProcessor(input_channels=input_shape[2]) for _ in range(12)
        ])
        
        # 4 opinion groups (each processes 3 leads)
        self.opinion_groups = nn.ModuleList([
            OpinionGroup(input_channels=96) for _ in range(4)  # 32 * 3 = 96
        ])
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc1 = nn.Linear(256, 64)  # 64 * 4 = 256 from 4 opinion groups
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, inputs: List[torch.Tensor]):
        """
        Forward pass with list of 12 lead inputs.
        
        Args:
            inputs: List of 12 tensors, each of shape (batch_size, channels, height, width)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        if len(inputs) != 12:
            raise ValueError(f"Expected 12 lead inputs, got {len(inputs)}")
        
        # Process each lead individually
        lead_outputs = []
        for i, lead_input in enumerate(inputs):
            processed = self.lead_processors[i](lead_input)
            lead_outputs.append(processed)
        
        # Group leads into 4 opinion groups (3 leads each)
        opinion_outputs = []
        for i in range(4):
            # Concatenate 3 leads along channel dimension
            group_leads = lead_outputs[i*3:(i+1)*3]
            concatenated = torch.cat(group_leads, dim=1)  # Concatenate along channel dim
            
            # Process the group
            group_output = self.opinion_groups[i](concatenated)
            opinion_outputs.append(group_output)
        
        # Merge all opinion groups
        merged = torch.cat(opinion_outputs, dim=1)  # Concatenate along channel dim
        
        # Global average pooling
        x = self.global_avg_pool(merged)
        x = torch.flatten(x, 1)
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x


class MultibranchCNN2DSingleInput(nn.Module):
    """
    Alternative implementation that takes a single tensor input and splits it into leads.
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        super(MultibranchCNN2DSingleInput, self).__init__()
        
        self.multibranch_model = MultibranchCNN2D(input_shape, num_classes)
    
    def forward(self, x):
        """
        Forward pass with single input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, 12, height, width) or (batch_size, height, width, 12)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Split the input into 12 leads
        if x.shape[1] == 12:  # (batch, 12, height, width)
            lead_inputs = [x[:, i:i+1, :, :] for i in range(12)]
        elif x.shape[-1] == 12:  # (batch, height, width, 12)
            # Convert to (batch, 12, height, width) format
            x = x.permute(0, 3, 1, 2)
            lead_inputs = [x[:, i:i+1, :, :] for i in range(12)]
        else:
            raise ValueError(f"Input shape {x.shape} not supported. Expected last dim or second dim to be 12.")
        
        return self.multibranch_model(lead_inputs)


def create_ecg_territory_moe(input_shape: Tuple[int, int, int], num_classes: int, 
                            single_input: bool = False) -> nn.Module:
    """
    Create a territory-based MoE model for 12-lead ECG classification.
    
    Args:
        input_shape: Tuple of (height, width, channels) for each lead
        num_classes: Number of output classes
        single_input: If True, model expects single tensor input, otherwise list of 12 tensors
    
    Returns:
        ECGTerritoryMoE model
    """
    if single_input:
        return ECGTerritoryMoESingleInput(input_shape, num_classes)
    else:
        return ECGTerritoryMoE(input_shape, num_classes)


def create_multibranch_cnn(input_shape: Tuple[int, int, int], num_classes: int, 
                          single_input: bool = False) -> nn.Module:
    """
    Create a multi-branch CNN model for 12-lead ECG classification.
    
    Args:
        input_shape: Tuple of (height, width, channels) for each lead
        num_classes: Number of output classes
        single_input: If True, model expects single tensor input, otherwise list of 12 tensors
    
    Returns:
        MultibranchCNN2D model
    """
    if single_input:
        return MultibranchCNN2DSingleInput(input_shape, num_classes)
    else:
        return MultibranchCNN2D(input_shape, num_classes)


def save_territory_moe_model(model: nn.Module, filepath: str, model_state_dict_only: bool = True):
    """
    Save the territory MoE model to the specified filepath.
    
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


def save_territory_moe_model_to_trained_dir(model: nn.Module, model_name: str, 
                                           models_dir: str = "models/trained"):
    """
    Save territory MoE model to the trained models directory with .keras extension.
    
    Args:
        model: PyTorch model to save
        model_name: Name for the model file (without extension)
        models_dir: Directory where to save the model
    """
    if not model_name.endswith('.keras'):
        model_name += '.keras'
    
    filepath = os.path.join(models_dir, model_name)
    save_territory_moe_model(model, filepath, model_state_dict_only=True)
    return filepath


def save_multibranch_model(model: nn.Module, filepath: str, model_state_dict_only: bool = True):
    """
    Save the multi-branch model to the specified filepath.
    
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


def save_multibranch_model_to_trained_dir(model: nn.Module, model_name: str, 
                                         models_dir: str = "models/trained"):
    """
    Save multi-branch model to the trained models directory with .keras extension.
    
    Args:
        model: PyTorch model to save
        model_name: Name for the model file (without extension)
        models_dir: Directory where to save the model
    """
    if not model_name.endswith('.keras'):
        model_name += '.keras'
    
    filepath = os.path.join(models_dir, model_name)
    save_multibranch_model(model, filepath, model_state_dict_only=True)
    return filepath


def load_territory_moe_model(filepath: str, model_class=None, **model_kwargs):
    """
    Load a saved territory MoE model.
    
    Args:
        filepath: Path to the saved model
        model_class: Model class to instantiate (required if loading state_dict)
        **model_kwargs: Arguments to pass to model constructor
    
    Returns:
        Loaded PyTorch model
    """
    if filepath.endswith('.keras') and model_class is not None:
        # Load state dict
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
        return model
    else:
        # Load complete model
        return torch.load(filepath, map_location='cpu')


def load_multibranch_model(filepath: str, model_class=None, **model_kwargs):
    """
    Load a saved multi-branch model.
    
    Args:
        filepath: Path to the saved model
        model_class: Model class to instantiate (required if loading state_dict)
        **model_kwargs: Arguments to pass to model constructor
    
    Returns:
        Loaded PyTorch model
    """
    if filepath.endswith('.keras') and model_class is not None:
        # Load state dict
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(filepath, map_location='cpu'))
        return model
    else:
        # Load complete model
        return torch.load(filepath, map_location='cpu')


def calculate_class_weights(y_train, num_classes: int, normal_class_idx: int = 3, 
                          mi_class_idx: int = 2) -> torch.Tensor:
    """
    Calculate class weights similar to the original TensorFlow implementation.
    
    Args:
        y_train: Training labels
        num_classes: Number of classes
        normal_class_idx: Index of the NORMAL class (default weight 0.4)
        mi_class_idx: Index of the MI class (default weight 0.8)
    
    Returns:
        Tensor of class weights
    """
    class_weights = torch.ones(num_classes)
    class_weights[normal_class_idx] = 0.4  # NORMAL class with lower weight
    class_weights[mi_class_idx] = 0.8      # MI class with lower weight
    return class_weights


# Example usage:
if __name__ == "__main__":
    # Example model creation
    input_shape = (33, 30, 1)  # (height, width, channels) for each ECG lead
    num_classes = 5
    
    print("🔥 Territory-based MoE Model for ECG Classification 🔥")
    print(f"Input shape per lead: {input_shape}")
    print(f"Number of classes: {num_classes}")
    
    # Create territory-based MoE model with list inputs (12 separate tensors)
    moe_model = create_ecg_territory_moe(input_shape, num_classes, single_input=False)
    print(f"\n📌 Territory MoE model (list input) created")
    print(f"Total parameters: {sum(p.numel() for p in moe_model.parameters()):,}")
    
    # Create territory-based MoE model with single input (single tensor)
    moe_single = create_ecg_territory_moe(input_shape, num_classes, single_input=True)
    print(f"\n📌 Territory MoE model (single input) created")
    
    # Print expert territories
    print("\n🎯 Expert Territories:")
    for territory, leads in moe_model.lead_territories.items():
        lead_names = []
        lead_mapping = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        for idx in leads:
            lead_names.append(lead_mapping[idx])
        print(f"  {territory.title()}: {', '.join(lead_names)} (indices: {leads})")
    
    # Example forward pass with list input
    batch_size = 4
    lead_inputs = [torch.randn(batch_size, 1, 33, 30) for _ in range(12)]
    with torch.no_grad():
        output_list = moe_model(lead_inputs)
    print(f"\n✅ List input test - Output shape: {output_list.shape}")
    
    # Example forward pass with single input
    single_input = torch.randn(batch_size, 12, 33, 30)  # 12 leads combined
    with torch.no_grad():
        output_single = moe_single(single_input)
    print(f"✅ Single input test - Output shape: {output_single.shape}")
    
    # Example save
    save_territory_moe_model_to_trained_dir(moe_single, "ecg_territory_moe_example")
    
    # Example class weights
    dummy_labels = torch.randint(0, num_classes, (1000,))
    weights = calculate_class_weights(dummy_labels, num_classes)
    print(f"\n⚖️ Class weights: {weights}")
    
    print("\n🚀 Territory-based MoE model ready for training!")
