import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from models import create_models
from dataset import visualize_samples, denormalize_tensor


def load_trained_model(checkpoint_path, noise_dim=100, num_classes=10, 
                      img_channels=1, feature_map_size=64, device='cpu'):
    """
    Load a trained conditional GAN model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        noise_dim: Dimension of noise vector
        num_classes: Number of classes
        img_channels: Number of image channels
        feature_map_size: Base feature map size
        device: Device to load model on
        
    Returns:
        generator: Loaded generator model
        checkpoint: Checkpoint dictionary with training info
    """
    # Create model
    generator, _ = create_models(noise_dim, num_classes, img_channels, feature_map_size)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    
    return generator, checkpoint


def generate_samples(generator, num_samples, labels, noise_dim=100, device='cpu'):
    """
    Generate samples using the trained generator.
    
    Args:
        generator: Trained generator model
        num_samples: Number of samples to generate
        labels: List of labels for conditional generation
        noise_dim: Dimension of noise vector
        device: Device to generate on
        
    Returns:
        generated_images: Tensor of generated images
        labels_tensor: Tensor of corresponding labels
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate random noise
        noise = torch.randn(num_samples, noise_dim, device=device)
        
        # Convert labels to tensor
        if isinstance(labels, (list, np.ndarray)):
            labels_tensor = torch.tensor(labels, device=device)
        else:
            labels_tensor = labels.to(device)
        
        # Generate images
        generated_images = generator(noise, labels_tensor)
    
    return generated_images, labels_tensor


def generate_class_samples(generator, class_label, num_samples=16, noise_dim=100, device='cpu'):
    """
    Generate multiple samples for a specific class.
    
    Args:
        generator: Trained generator model
        class_label: Class label to generate
        num_samples: Number of samples to generate
        noise_dim: Dimension of noise vector
        device: Device to generate on
        
    Returns:
        generated_images: Tensor of generated images
        labels: Tensor of labels (all same class)
    """
    labels = [class_label] * num_samples
    return generate_samples(generator, num_samples, labels, noise_dim, device)


def generate_grid_samples(generator, num_classes=10, samples_per_class=8, 
                         noise_dim=100, device='cpu'):
    """
    Generate a grid of samples with equal representation from all classes.
    
    Args:
        generator: Trained generator model
        num_classes: Number of classes
        samples_per_class: Number of samples per class
        noise_dim: Dimension of noise vector
        device: Device to generate on
        
    Returns:
        generated_images: Tensor of generated images
        labels: Tensor of corresponding labels
    """
    total_samples = num_classes * samples_per_class
    labels = []
    
    # Create labels with equal representation
    for class_idx in range(num_classes):
        labels.extend([class_idx] * samples_per_class)
    
    return generate_samples(generator, total_samples, labels, noise_dim, device)


def save_generated_samples(images, labels, save_path, title="Generated Samples"):
    """
    Save generated samples as an image file.
    
    Args:
        images: Tensor of generated images
        labels: Corresponding labels
        save_path: Path to save the image
        title: Title for the plot
    """
    fig = visualize_samples(images, labels, num_samples=len(images), title=title)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved generated samples to {save_path}")


def interpolate_between_classes(generator, class1, class2, num_steps=10, 
                               noise_dim=100, device='cpu'):
    """
    Generate interpolation between two classes.
    
    Args:
        generator: Trained generator model
        class1: First class label
        class2: Second class label
        num_steps: Number of interpolation steps
        noise_dim: Dimension of noise vector
        device: Device to generate on
        
    Returns:
        interpolated_images: Tensor of interpolated images
        interpolation_labels: Labels for visualization
    """
    generator.eval()
    
    with torch.no_grad():
        # Fixed noise for consistent comparison
        noise = torch.randn(1, noise_dim, device=device)
        noise = noise.repeat(num_steps, 1)
        
        # Get embeddings for both classes
        label1_tensor = torch.tensor([class1], device=device)
        label2_tensor = torch.tensor([class2], device=device)
        
        embedding1 = generator.label_embedding(label1_tensor)
        embedding2 = generator.label_embedding(label2_tensor)
        
        # Generate interpolation steps
        interpolated_images = []
        interpolation_labels = []
        
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            
            # Interpolate between embeddings
            interpolated_embedding = (1 - alpha) * embedding1 + alpha * embedding2
            
            # Generate input
            gen_input = torch.cat([noise[i:i+1], interpolated_embedding], dim=1)
            
            # Transform to initial feature maps and generate image
            x = generator.initial_dense(gen_input)
            x = x.view(-1, generator.feature_map_size * 8, 7, 7)
            
            # Pass through upsampling layers
            x = generator.upsample1(x)
            x = generator.conv1(x)
            x = generator.bn1(x)
            x = torch.relu(x)
            
            x = generator.upsample2(x)
            x = generator.conv2(x)
            x = generator.bn2(x)
            x = torch.relu(x)
            
            x = generator.conv3(x)
            x = generator.bn3(x)
            x = torch.relu(x)
            
            x = generator.conv4(x)
            x = torch.tanh(x)
            
            interpolated_images.append(x)
            interpolation_labels.append(f"{class1}→{class2} ({alpha:.2f})")
    
    interpolated_images = torch.cat(interpolated_images, dim=0)
    
    return interpolated_images, interpolation_labels


def main():
    """
    Main inference function with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Generate samples with trained Conditional GAN')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                       help='Directory to save generated samples')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--class_label', type=int, default=None,
                       help='Specific class to generate (0-9), if not specified generates from all classes')
    parser.add_argument('--noise_dim', type=int, default=100,
                       help='Dimension of noise vector')
    parser.add_argument('--feature_map_size', type=int, default=64,
                       help='Base feature map size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate interpolation between classes')
    parser.add_argument('--interpolate_classes', type=int, nargs=2, default=[0, 9],
                       help='Two classes to interpolate between')
    parser.add_argument('--interpolate_steps', type=int, default=10,
                       help='Number of interpolation steps')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained model
    print(f"Loading model from {args.checkpoint}...")
    generator, checkpoint = load_trained_model(
        args.checkpoint, 
        noise_dim=args.noise_dim,
        feature_map_size=args.feature_map_size,
        device=device
    )
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Final Generator Loss: {checkpoint.get('g_loss', 'N/A')}")
    print(f"Final Discriminator Loss: {checkpoint.get('d_loss', 'N/A')}")
    
    if args.interpolate:
        # Generate interpolation between classes
        print(f"Generating interpolation between classes {args.interpolate_classes[0]} and {args.interpolate_classes[1]}...")
        
        interpolated_images, interpolation_labels = interpolate_between_classes(
            generator,
            args.interpolate_classes[0],
            args.interpolate_classes[1],
            num_steps=args.interpolate_steps,
            noise_dim=args.noise_dim,
            device=device
        )
        
        # Create custom visualization for interpolation
        fig, axes = plt.subplots(1, args.interpolate_steps, figsize=(2*args.interpolate_steps, 2))
        fig.suptitle(f'Interpolation: Class {args.interpolate_classes[0]} → Class {args.interpolate_classes[1]}', fontsize=14)
        
        for i in range(args.interpolate_steps):
            img = denormalize_tensor(interpolated_images[i]).squeeze().cpu().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Step {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(args.output_dir, f'interpolation_{args.interpolate_classes[0]}_to_{args.interpolate_classes[1]}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved interpolation to {save_path}")
        
    else:
        if args.class_label is not None:
            # Generate samples for specific class
            print(f"Generating {args.num_samples} samples for class {args.class_label}...")
            
            generated_images, labels = generate_class_samples(
                generator,
                args.class_label,
                num_samples=args.num_samples,
                noise_dim=args.noise_dim,
                device=device
            )
            
            save_path = os.path.join(args.output_dir, f'class_{args.class_label}_samples.png')
            save_generated_samples(
                generated_images, 
                labels, 
                save_path,
                title=f'Generated Samples - Class {args.class_label}'
            )
            
        else:
            # Generate grid with all classes
            print(f"Generating samples from all classes...")
            
            samples_per_class = max(1, args.num_samples // 10)
            generated_images, labels = generate_grid_samples(
                generator,
                num_classes=10,
                samples_per_class=samples_per_class,
                noise_dim=args.noise_dim,
                device=device
            )
            
            save_path = os.path.join(args.output_dir, 'all_classes_samples.png')
            save_generated_samples(
                generated_images, 
                labels, 
                save_path,
                title='Generated Samples - All Classes'
            )
    
    print("Inference completed!")


if __name__ == '__main__':
    main()