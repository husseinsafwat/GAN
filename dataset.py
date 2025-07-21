import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


def get_mnist_dataloader(batch_size=64, download=True, data_dir='./data'):
    """
    Create MNIST dataloader with appropriate preprocessing.
    
    Args:
        batch_size: Batch size for training
        download: Whether to download MNIST if not available
        data_dir: Directory to store/load MNIST data
        
    Returns:
        dataloader: PyTorch DataLoader for MNIST
        dataset: The MNIST dataset object
    """
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] range
    ])
    
    # Load MNIST dataset
    dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=download,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    return dataloader, dataset


def denormalize_tensor(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1] range for visualization.
    
    Args:
        tensor: Normalized tensor in [-1, 1] range
        
    Returns:
        Denormalized tensor in [0, 1] range
    """
    return (tensor + 1) / 2


def visualize_samples(images, labels, num_samples=16, figsize=(8, 8), title="Samples"):
    """
    Visualize a grid of image samples with their labels.
    
    Args:
        images: Tensor of images to visualize
        labels: Corresponding labels
        num_samples: Number of samples to show
        figsize: Figure size for matplotlib
        title: Title for the plot
    """
    # Ensure we don't exceed available samples
    num_samples = min(num_samples, len(images))
    
    # Create subplot grid
    grid_size = int(np.sqrt(num_samples))
    if grid_size * grid_size < num_samples:
        grid_size += 1
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    fig.suptitle(title, fontsize=16)
    
    for i in range(num_samples):
        row = i // grid_size
        col = i % grid_size
        
        if grid_size == 1:
            ax = axes
        elif grid_size == 2:
            ax = axes[col] if num_samples <= 2 else axes[row, col]
        else:
            ax = axes[row, col]
        
        # Denormalize and convert to numpy
        img = denormalize_tensor(images[i]).squeeze().cpu().numpy()
        label = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Label: {label}')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_samples, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        if grid_size > 1:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig


def generate_fixed_noise_and_labels(batch_size, noise_dim, num_classes, device):
    """
    Generate fixed noise and labels for consistent visualization during training.
    
    Args:
        batch_size: Number of samples to generate
        noise_dim: Dimension of noise vector
        num_classes: Number of classes
        device: Device to put tensors on
        
    Returns:
        fixed_noise, fixed_labels: Fixed tensors for visualization
    """
    # Generate fixed noise
    fixed_noise = torch.randn(batch_size, noise_dim, device=device)
    
    # Generate fixed labels (evenly distributed across classes)
    samples_per_class = batch_size // num_classes
    fixed_labels = []
    
    for class_idx in range(num_classes):
        fixed_labels.extend([class_idx] * samples_per_class)
    
    # Fill remaining slots if batch_size is not divisible by num_classes
    while len(fixed_labels) < batch_size:
        fixed_labels.append(len(fixed_labels) % num_classes)
    
    fixed_labels = torch.tensor(fixed_labels[:batch_size], device=device)
    
    return fixed_noise, fixed_labels


def save_sample_grid(generator, fixed_noise, fixed_labels, epoch, save_dir='./samples'):
    """
    Generate and save a grid of samples from the generator.
    
    Args:
        generator: Trained generator model
        fixed_noise: Fixed noise tensor
        fixed_labels: Fixed labels tensor
        epoch: Current epoch number
        save_dir: Directory to save samples
    """
    import os
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate samples
    with torch.no_grad():
        fake_images = generator(fixed_noise, fixed_labels)
    
    # Create and save visualization
    fig = visualize_samples(
        fake_images, 
        fixed_labels, 
        num_samples=len(fake_images),
        title=f'Generated Samples - Epoch {epoch}'
    )
    
    plt.savefig(f'{save_dir}/samples_epoch_{epoch:04d}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def get_class_distribution(dataloader):
    """
    Get the distribution of classes in the dataset.
    
    Args:
        dataloader: DataLoader for the dataset
        
    Returns:
        class_counts: Dictionary with class counts
    """
    class_counts = {}
    
    for _, labels in dataloader:
        for label in labels:
            label_item = label.item()
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
    
    return class_counts