import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import argparse
import os
from tqdm import tqdm

from models import create_models
from dataset import get_mnist_dataloader, denormalize_tensor, visualize_samples
from inference import load_trained_model, generate_grid_samples


class MNISTClassifier(nn.Module):
    """
    Simple CNN classifier for evaluating generated MNIST samples.
    """
    
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def train_classifier(device='cpu', epochs=10, batch_size=64):
    """
    Train a classifier on real MNIST data for evaluation purposes.
    
    Args:
        device: Device to train on
        epochs: Number of training epochs
        batch_size: Training batch size
        
    Returns:
        classifier: Trained classifier model
    """
    print("Training MNIST classifier for evaluation...")
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train classifier
    classifier = MNISTClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluate on test set
    classifier.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    return classifier


def calculate_inception_score(generator, classifier, num_samples=10000, 
                            batch_size=100, num_classes=10, noise_dim=100, 
                            device='cpu'):
    """
    Calculate Inception Score using a trained classifier.
    
    Args:
        generator: Trained generator model
        classifier: Trained classifier for evaluation
        num_samples: Number of samples to generate for evaluation
        batch_size: Batch size for generation
        num_classes: Number of classes
        noise_dim: Dimension of noise vector
        device: Device to use
        
    Returns:
        inception_score: Calculated inception score
        std: Standard deviation of inception score
    """
    print(f"Calculating Inception Score with {num_samples} samples...")
    
    generator.eval()
    classifier.eval()
    
    # Collect predictions
    all_predictions = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating samples for IS"):
            current_batch_size = min(batch_size, num_samples - len(all_predictions) * batch_size)
            if current_batch_size <= 0:
                break
            
            # Generate random labels
            labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            
            # Generate samples
            fake_images = generator(noise, labels)
            
            # Get predictions
            outputs = classifier(fake_images)
            predictions = torch.softmax(outputs, dim=1)
            all_predictions.append(predictions.cpu())
    
    # Concatenate all predictions
    all_predictions = torch.cat(all_predictions, dim=0)[:num_samples]
    
    # Calculate inception score
    # IS = exp(E[KL(p(y|x) || p(y))])
    
    # Calculate marginal distribution p(y)
    marginal = torch.mean(all_predictions, dim=0)
    
    # Calculate KL divergence for each sample
    kl_divs = []
    for i in range(all_predictions.shape[0]):
        p_yx = all_predictions[i]
        kl_div = torch.sum(p_yx * torch.log(p_yx / marginal + 1e-8))
        kl_divs.append(kl_div.item())
    
    # Calculate mean and std of KL divergences
    mean_kl = np.mean(kl_divs)
    std_kl = np.std(kl_divs)
    
    # Inception score is exp(mean_kl)
    inception_score = np.exp(mean_kl)
    
    return inception_score, std_kl


def calculate_fid_score(real_features, fake_features):
    """
    Calculate Fréchet Inception Distance (FID) score.
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        
    Returns:
        fid_score: Calculated FID score
    """
    # Calculate means
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    # Calculate covariance matrices
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_fake
    
    # Product might be almost singular
    covmean, _ = np.linalg.eig(sigma_real.dot(sigma_fake))
    covmean = np.sqrt(np.abs(covmean)).real
    
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.sum(covmean)
    
    return fid


def extract_features(model, dataloader, device='cpu'):
    """
    Extract features from images using a pre-trained model.
    
    Args:
        model: Feature extraction model
        dataloader: DataLoader with images
        device: Device to use
        
    Returns:
        features: Extracted features
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Extract features (remove the last classification layer)
            x = model.conv_layers(images)
            x = torch.flatten(x, 1)
            features.append(x.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def evaluate_class_diversity(generator, num_samples_per_class=1000, 
                           noise_dim=100, device='cpu'):
    """
    Evaluate diversity of generated samples for each class.
    
    Args:
        generator: Trained generator model
        num_samples_per_class: Number of samples to generate per class
        noise_dim: Dimension of noise vector
        device: Device to use
        
    Returns:
        diversity_scores: Dictionary with diversity scores per class
    """
    print("Evaluating class diversity...")
    
    generator.eval()
    diversity_scores = {}
    
    with torch.no_grad():
        for class_label in range(10):
            # Generate samples for this class
            labels = torch.full((num_samples_per_class,), class_label, device=device)
            noise = torch.randn(num_samples_per_class, noise_dim, device=device)
            
            generated_images = generator(noise, labels)
            
            # Flatten images for diversity calculation
            flattened_images = generated_images.view(num_samples_per_class, -1).cpu().numpy()
            
            # Calculate pairwise distances
            distances = []
            for i in range(num_samples_per_class):
                for j in range(i + 1, num_samples_per_class):
                    dist = np.linalg.norm(flattened_images[i] - flattened_images[j])
                    distances.append(dist)
            
            # Average distance as diversity measure
            diversity_scores[class_label] = np.mean(distances)
    
    return diversity_scores


def evaluate_conditional_accuracy(generator, classifier, num_samples=10000,
                                 batch_size=100, noise_dim=100, device='cpu'):
    """
    Evaluate how well the generator follows the conditional labels.
    
    Args:
        generator: Trained generator model
        classifier: Trained classifier
        num_samples: Number of samples to evaluate
        batch_size: Batch size for evaluation
        noise_dim: Dimension of noise vector
        device: Device to use
        
    Returns:
        accuracy: Classification accuracy of generated samples
        class_accuracies: Per-class accuracies
    """
    print(f"Evaluating conditional accuracy with {num_samples} samples...")
    
    generator.eval()
    classifier.eval()
    
    all_predicted = []
    all_target = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Evaluating conditional accuracy"):
            current_batch_size = min(batch_size, num_samples - len(all_predicted) * batch_size)
            if current_batch_size <= 0:
                break
            
            # Generate samples with specific labels
            target_labels = torch.randint(0, 10, (current_batch_size,), device=device)
            noise = torch.randn(current_batch_size, noise_dim, device=device)
            
            generated_images = generator(noise, target_labels)
            
            # Classify generated images
            outputs = classifier(generated_images)
            _, predicted_labels = torch.max(outputs, 1)
            
            all_predicted.extend(predicted_labels.cpu().numpy())
            all_target.extend(target_labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = accuracy_score(all_target, all_predicted)
    
    # Calculate per-class accuracies
    class_report = classification_report(all_target, all_predicted, 
                                       target_names=[str(i) for i in range(10)],
                                       output_dict=True)
    
    return accuracy, class_report


def main():
    """
    Main evaluation function.
    """
    parser = argparse.ArgumentParser(description='Evaluate Conditional GAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to generator checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--noise_dim', type=int, default=100,
                       help='Dimension of noise vector')
    parser.add_argument('--feature_map_size', type=int, default=64,
                       help='Base feature map size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples for evaluation')
    parser.add_argument('--batch_size', type=int, default=100,
                       help='Batch size for evaluation')
    parser.add_argument('--train_classifier', action='store_true',
                       help='Train a new classifier (otherwise tries to load existing one)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load generator
    print(f"Loading generator from {args.checkpoint}...")
    generator, checkpoint = load_trained_model(
        args.checkpoint,
        noise_dim=args.noise_dim,
        feature_map_size=args.feature_map_size,
        device=device
    )
    
    # Load or train classifier
    classifier_path = os.path.join(args.output_dir, 'mnist_classifier.pth')
    
    if args.train_classifier or not os.path.exists(classifier_path):
        classifier = train_classifier(device=device)
        torch.save(classifier.state_dict(), classifier_path)
        print(f"Saved classifier to {classifier_path}")
    else:
        print(f"Loading existing classifier from {classifier_path}")
        classifier = MNISTClassifier().to(device)
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    # Generate sample visualization
    print("Generating sample visualization...")
    sample_images, sample_labels = generate_grid_samples(
        generator, num_classes=10, samples_per_class=8,
        noise_dim=args.noise_dim, device=device
    )
    
    fig = visualize_samples(sample_images, sample_labels, 
                          title="Generated Samples for Evaluation")
    plt.savefig(os.path.join(args.output_dir, 'evaluation_samples.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # Evaluate conditional accuracy
    accuracy, class_report = evaluate_conditional_accuracy(
        generator, classifier, num_samples=args.num_samples,
        batch_size=args.batch_size, noise_dim=args.noise_dim, device=device
    )
    
    print(f"\nConditional Accuracy: {accuracy:.4f}")
    print("\nPer-class Performance:")
    for class_idx in range(10):
        class_metrics = class_report[str(class_idx)]
        print(f"Class {class_idx}: Precision={class_metrics['precision']:.3f}, "
              f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")
    
    # Calculate Inception Score
    inception_score, inception_std = calculate_inception_score(
        generator, classifier, num_samples=args.num_samples,
        batch_size=args.batch_size, noise_dim=args.noise_dim, device=device
    )
    
    print(f"\nInception Score: {inception_score:.4f} ± {inception_std:.4f}")
    
    # Evaluate class diversity
    diversity_scores = evaluate_class_diversity(
        generator, num_samples_per_class=1000, 
        noise_dim=args.noise_dim, device=device
    )
    
    print("\nClass Diversity Scores:")
    for class_idx, score in diversity_scores.items():
        print(f"Class {class_idx}: {score:.4f}")
    
    avg_diversity = np.mean(list(diversity_scores.values()))
    print(f"Average Diversity: {avg_diversity:.4f}")
    
    # Save evaluation results
    results = {
        'conditional_accuracy': accuracy,
        'class_report': class_report,
        'inception_score': inception_score,
        'inception_std': inception_std,
        'diversity_scores': diversity_scores,
        'average_diversity': avg_diversity,
        'checkpoint_epoch': checkpoint['epoch'],
        'generator_loss': checkpoint.get('g_loss', 'N/A'),
        'discriminator_loss': checkpoint.get('d_loss', 'N/A')
    }
    
    # Save results as text file
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Conditional GAN Evaluation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Training Epochs: {results['checkpoint_epoch']}\n")
        f.write(f"Generator Loss: {results['generator_loss']}\n")
        f.write(f"Discriminator Loss: {results['discriminator_loss']}\n\n")
        
        f.write(f"Conditional Accuracy: {results['conditional_accuracy']:.4f}\n\n")
        
        f.write("Per-class Performance:\n")
        for class_idx in range(10):
            if str(class_idx) in results['class_report']:
                class_metrics = results['class_report'][str(class_idx)]
                f.write(f"Class {class_idx}: Precision={class_metrics['precision']:.3f}, "
                       f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}\n")
        
        f.write(f"\nInception Score: {results['inception_score']:.4f} ± {results['inception_std']:.4f}\n\n")
        
        f.write("Class Diversity Scores:\n")
        for class_idx, score in results['diversity_scores'].items():
            f.write(f"Class {class_idx}: {score:.4f}\n")
        f.write(f"Average Diversity: {results['average_diversity']:.4f}\n")
    
    print(f"\nEvaluation results saved to {args.output_dir}")
    print("Evaluation completed!")


if __name__ == '__main__':
    main()