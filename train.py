import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from models import create_models
from dataset import (
    get_mnist_dataloader, 
    generate_fixed_noise_and_labels, 
    save_sample_grid,
    visualize_samples
)


def train_conditional_gan(
    generator, 
    discriminator, 
    dataloader, 
    device, 
    num_epochs=100,
    lr_g=0.0002, 
    lr_d=0.0002,
    beta1=0.5,
    beta2=0.999,
    noise_dim=100,
    num_classes=10,
    save_dir='./checkpoints',
    sample_dir='./samples',
    log_dir='./logs',
    save_interval=10,
    sample_interval=5
):
    """
    Train the conditional GAN model.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        dataloader: Training data loader
        device: Device to train on
        num_epochs: Number of training epochs
        lr_g: Learning rate for generator
        lr_d: Learning rate for discriminator
        beta1, beta2: Adam optimizer parameters
        noise_dim: Dimension of noise vector
        num_classes: Number of classes
        save_dir: Directory to save model checkpoints
        sample_dir: Directory to save generated samples
        log_dir: Directory for tensorboard logs
        save_interval: Interval to save model checkpoints
        sample_interval: Interval to save sample images
    """
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    
    # Generate fixed noise and labels for consistent visualization
    fixed_noise, fixed_labels = generate_fixed_noise_and_labels(
        64, noise_dim, num_classes, device
    )
    
    # Training statistics
    g_losses = []
    d_losses = []
    
    print(f"Starting training on {device}...")
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0
        
        # Training progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (real_images, real_labels) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            real_labels = real_labels.to(device)
            
            # Create labels for real/fake discrimination
            real_target = torch.ones(batch_size, 1, device=device)
            fake_target = torch.zeros(batch_size, 1, device=device)
            
            # =====================================
            # Train Discriminator
            # =====================================
            
            optimizer_d.zero_grad()
            
            # Train with real images
            output_real = discriminator(real_images, real_labels)
            d_loss_real = criterion(output_real, real_target)
            
            # Generate fake images
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_images = generator(noise, fake_labels)
            
            # Train with fake images
            output_fake = discriminator(fake_images.detach(), fake_labels)
            d_loss_fake = criterion(output_fake, fake_target)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()
            
            # =====================================
            # Train Generator
            # =====================================
            
            optimizer_g.zero_grad()
            
            # Generate fake images and get discriminator output
            output = discriminator(fake_images, fake_labels)
            g_loss = criterion(output, real_target)  # Want discriminator to think they're real
            
            g_loss.backward()
            optimizer_g.step()
            
            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'D_Loss': f'{d_loss.item():.4f}',
                'G_Loss': f'{g_loss.item():.4f}'
            })
            
            # Log to tensorboard (every 100 batches)
            if batch_idx % 100 == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('Loss/Discriminator_Batch', d_loss.item(), global_step)
                writer.add_scalar('Loss/Generator_Batch', g_loss.item(), global_step)
                writer.add_scalar('Discriminator/Real_Output', output_real.mean().item(), global_step)
                writer.add_scalar('Discriminator/Fake_Output', output_fake.mean().item(), global_step)
        
        # Calculate average losses for the epoch
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Log epoch statistics
        writer.add_scalar('Loss/Generator_Epoch', avg_g_loss, epoch)
        writer.add_scalar('Loss/Discriminator_Epoch', avg_d_loss, epoch)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - G_Loss: {avg_g_loss:.4f}, D_Loss: {avg_d_loss:.4f}')
        
        # Save sample images
        if (epoch + 1) % sample_interval == 0:
            save_sample_grid(generator, fixed_noise, fixed_labels, epoch + 1, sample_dir)
        
        # Save model checkpoints
        if (epoch + 1) % save_interval == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1:04d}.pth'))
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'g_losses': g_losses,
        'd_losses': d_losses,
    }, os.path.join(save_dir, 'final_model.pth'))
    
    # Plot and save training curves
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()
    
    writer.close()
    print("Training completed!")
    
    return g_losses, d_losses


def main():
    """
    Main training function with command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train Conditional GAN on MNIST')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=0.0002, help='Learning rate for discriminator')
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--feature_map_size', type=int, default=64, help='Base feature map size')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--sample_interval', type=int, default=5, help='Sample generation interval')
    parser.add_argument('--device', type=str, default='auto', help='Device to train on (auto/cpu/cuda)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for MNIST data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save models')
    parser.add_argument('--sample_dir', type=str, default='./samples', help='Directory to save samples')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading MNIST dataset...")
    dataloader, dataset = get_mnist_dataloader(
        batch_size=args.batch_size, 
        data_dir=args.data_dir
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Create models
    print("Creating models...")
    generator, discriminator = create_models(
        noise_dim=args.noise_dim,
        num_classes=10,
        img_channels=1,
        feature_map_size=args.feature_map_size
    )
    
    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Train the model
    g_losses, d_losses = train_conditional_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        num_epochs=args.epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        noise_dim=args.noise_dim,
        save_dir=args.save_dir,
        sample_dir=args.sample_dir,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval
    )


if __name__ == '__main__':
    main()