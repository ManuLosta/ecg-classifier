from src.data.ptbxl_loader import PTBXLDataset
import matplotlib.pyplot as plt
import numpy as np

def main():
    data_path = 'dataset'
    ptbxl_dataset = PTBXLDataset(data_path)

    # Get the dataset splits
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_names = ptbxl_dataset.get_dataset()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Calculate class distribution in training data
    class_counts = np.sum(y_train, axis=0)
    class_percentages = (class_counts / len(y_train)) * 100
    
    # Create bar plot for class distribution
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(class_names)), class_counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Diagnostic Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Distribution of Diagnostic Classes in Training Data', fontsize=14, fontweight='bold')
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    
    # Add value labels on top of bars
    for i, (bar, count, percentage) in enumerate(zip(bars, class_counts, class_percentages)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    # Print detailed statistics
    print("\nClass Distribution Statistics:")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_counts[i]} samples ({class_percentages[i]:.2f}%)")

if __name__ == "__main__":
    main()
