#!/usr/bin/env python3
"""
Quick Example: Conditional GAN for MNIST
=========================================

This script demonstrates the basic usage of the conditional GAN implementation.
It shows how to create models, train for a few epochs, and generate samples.

This is a simplified example for demonstration purposes.
For full training, use train.py with appropriate parameters.
"""

import torch
import matplotlib.pyplot as plt
from models import create_models
from dataset import get_mnist_dataloader, visualize_samples, generate_fixed_noise_and_labels
import os


def quick_demo():
    """
    Quick demonstration of the conditional GAN.
    """
    print("🚀 Conditional GAN Quick Demo")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    
    # Create models
    print("🏗️  Creating models...")
    generator, discriminator = create_models(
        noise_dim=100,
        num_classes=10,
        img_channels=1,
        feature_map_size=64
    )
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    print(f"✅ Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"✅ Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Load a small batch of data for demonstration
    print("📊 Loading MNIST data...")
    dataloader, dataset = get_mnist_dataloader(batch_size=16, download=True)
    print(f"✅ Dataset size: {len(dataset)}")
    
    # Show some real samples
    print("🖼️  Visualizing real samples...")
    real_batch = next(iter(dataloader))
    real_images, real_labels = real_batch
    
    fig = visualize_samples(real_images, real_labels, title="Real MNIST Samples")
    plt.savefig('real_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Real samples saved to 'real_samples.png'")
    
    # Generate samples with untrained model (should be noise)
    print("🎲 Generating samples with untrained model...")
    generator.eval()
    
    with torch.no_grad():
        # Generate fixed noise and labels for consistent comparison
        fixed_noise, fixed_labels = generate_fixed_noise_and_labels(
            16, 100, 10, device
        )
        
        # Generate samples
        fake_images = generator(fixed_noise, fixed_labels)
        
        # Visualize
        fig = visualize_samples(
            fake_images, 
            fixed_labels, 
            title="Generated Samples (Untrained Model)"
        )
        plt.savefig('untrained_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Untrained samples saved to 'untrained_samples.png'")
    
    # Demonstrate conditional generation
    print("🎯 Demonstrating conditional generation...")
    
    with torch.no_grad():
        # Generate specific digits (0-9)
        specific_labels = torch.arange(10, device=device)
        specific_noise = torch.randn(10, 100, device=device)
        
        specific_images = generator(specific_noise, specific_labels)
        
        fig = visualize_samples(
            specific_images,
            specific_labels,
            title="Conditional Generation (Digits 0-9, Untrained)"
        )
        plt.savefig('conditional_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Conditional samples saved to 'conditional_samples.png'")
    
    # Mini training demonstration (just a few steps)
    print("🏋️  Mini training demonstration (5 steps)...")
    
    criterion = torch.nn.BCELoss()
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.train()
    discriminator.train()
    
    # Get a few batches for mini training
    data_iter = iter(dataloader)
    
    for step in range(5):
        try:
            real_images, real_labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            real_images, real_labels = next(data_iter)
        
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        real_labels = real_labels.to(device)
        
        # Labels for real/fake
        real_target = torch.ones(batch_size, 1, device=device)
        fake_target = torch.zeros(batch_size, 1, device=device)
        
        # Train Discriminator
        optimizer_d.zero_grad()
        
        # Real samples
        output_real = discriminator(real_images, real_labels)
        d_loss_real = criterion(output_real, real_target)
        
        # Fake samples
        noise = torch.randn(batch_size, 100, device=device)
        fake_labels = torch.randint(0, 10, (batch_size,), device=device)
        fake_images = generator(noise, fake_labels)
        output_fake = discriminator(fake_images.detach(), fake_labels)
        d_loss_fake = criterion(output_fake, fake_target)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_d.step()
        
        # Train Generator
        optimizer_g.zero_grad()
        output = discriminator(fake_images, fake_labels)
        g_loss = criterion(output, real_target)
        g_loss.backward()
        optimizer_g.step()
        
        print(f"  Step {step+1}/5 - D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")
    
    # Generate samples after mini training
    print("🖼️  Generating samples after mini training...")
    generator.eval()
    
    with torch.no_grad():
        trained_images = generator(fixed_noise, fixed_labels)
        
        fig = visualize_samples(
            trained_images,
            fixed_labels,
            title="Generated Samples (After 5 Training Steps)"
        )
        plt.savefig('mini_trained_samples.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Mini-trained samples saved to 'mini_trained_samples.png'")
    
    print("\n🎉 Demo completed!")
    print("=" * 50)
    print("📁 Generated files:")
    print("  - real_samples.png: Real MNIST samples")
    print("  - untrained_samples.png: Samples from untrained model")
    print("  - conditional_samples.png: Conditional generation demo")
    print("  - mini_trained_samples.png: Samples after brief training")
    print("\n💡 For full training, run:")
    print("  python train.py --epochs 100")
    print("\n📚 See README.md for detailed usage instructions")


if __name__ == "__main__":
    quick_demo()