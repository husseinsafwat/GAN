# Conditional GAN for MNIST Dataset

A PyTorch implementation of Conditional Generative Adversarial Networks (GANs) for generating MNIST digit images. This implementation uses **Conv2D upsampling layers followed by kernel filters** instead of the conventional Conv2DTranspose layers in the generator architecture.

## Architecture Highlights

### Generator Architecture
- **Unique Design**: Uses Conv2D upsampling layers with nearest neighbor upsampling followed by 3x3 convolution kernels instead of Conv2DTranspose
- **4 Upsampling Layers**: 
  1. Dense layer → 7×7 feature maps (512 channels)
  2. Upsample + Conv2D: 7×7 → 14×14 (256 channels)
  3. Upsample + Conv2D: 14×14 → 28×28 (128 channels)  
  4. Refinement Conv2D: 28×28 → 28×28 (64 channels)
  5. Final Conv2D: 28×28 → 28×28 (1 channel, output)
- **Conditional Input**: Concatenates noise vector with embedded class labels
- **Batch Normalization**: Applied after each upsampling layer for stable training

### Discriminator Architecture
- **Conditional Discrimination**: Takes both image and class label as input
- **Label Embedding**: Class labels are embedded and reshaped to match image dimensions
- **Progressive Downsampling**: 28×28 → 14×14 → 7×7 → 3×3 → 1×1
- **Feature Extraction**: 64 → 128 → 256 → 512 channels

## Features

- ✅ **Conditional Generation**: Generate specific digit classes (0-9)
- ✅ **Custom Architecture**: Conv2D upsampling instead of Conv2DTranspose
- ✅ **Comprehensive Training**: Full training pipeline with monitoring
- ✅ **Advanced Evaluation**: Inception Score, conditional accuracy, diversity metrics
- ✅ **Visualization Tools**: Sample grids, training progress, interpolations
- ✅ **TensorBoard Integration**: Real-time training monitoring
- ✅ **Flexible Inference**: Generate specific classes or interpolate between classes

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch >= 2.0.0
- TorchVision >= 0.15.0
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- TensorBoard >= 2.10.0
- TQDM >= 4.64.0
- scikit-learn (for evaluation)

## Quick Start

### 1. Training
```bash
# Basic training (100 epochs, auto device detection)
python train.py

# Advanced training with custom parameters
python train.py --epochs 200 --batch_size 128 --lr_g 0.0001 --lr_d 0.0002 --device cuda
```

### 2. Generate Samples
```bash
# Generate samples from all classes using trained model
python inference.py --checkpoint ./checkpoints/final_model.pth

# Generate specific class samples
python inference.py --checkpoint ./checkpoints/final_model.pth --class_label 7 --num_samples 64

# Generate interpolation between classes
python inference.py --checkpoint ./checkpoints/final_model.pth --interpolate --interpolate_classes 0 9
```

### 3. Evaluate Model
```bash
# Comprehensive evaluation including Inception Score and conditional accuracy
python evaluate.py --checkpoint ./checkpoints/final_model.pth --num_samples 10000
```

## Usage Examples

### Training with Custom Parameters
```python
python train.py \
  --epochs 150 \
  --batch_size 64 \
  --lr_g 0.0002 \
  --lr_d 0.0002 \
  --noise_dim 100 \
  --feature_map_size 64 \
  --save_interval 10 \
  --sample_interval 5 \
  --device cuda
```

### Programmatic Usage
```python
from models import create_models
from dataset import get_mnist_dataloader
import torch

# Create models
generator, discriminator = create_models(
    noise_dim=100, 
    num_classes=10,
    feature_map_size=64
)

# Load data
dataloader, dataset = get_mnist_dataloader(batch_size=64)

# Generate samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noise = torch.randn(16, 100, device=device)
labels = torch.randint(0, 10, (16,), device=device)

generator.eval()
with torch.no_grad():
    generated_images = generator(noise, labels)
```

## Project Structure

```
conditional-gan-mnist/
├── models.py              # Generator and Discriminator architectures
├── dataset.py             # Data loading and visualization utilities
├── train.py               # Training script with command line interface
├── inference.py           # Inference and sample generation
├── evaluate.py            # Comprehensive model evaluation
├── requirements.txt       # Project dependencies
├── README.md              # This file
├── ARCHITECTURE.md        # Detailed architecture documentation
│
├── data/                  # MNIST dataset (auto-downloaded)
├── checkpoints/           # Model checkpoints and training curves
├── samples/               # Generated sample images during training
├── logs/                  # TensorBoard logs
├── generated_samples/     # Inference outputs
└── evaluation_results/    # Evaluation metrics and reports
```

## Key Architecture Differences

### Traditional DCGAN Generator vs Our Implementation

| Component | Traditional DCGAN | Our Implementation |
|-----------|-------------------|-------------------|
| Upsampling | Conv2DTranspose | Upsample + Conv2D |
| Kernel Size | 4×4 transposed conv | 2× nearest + 3×3 conv |
| Advantages | Direct learnable upsampling | Better control over upsampling artifacts |
| Parameters | Fewer parameters | More explicit control |

### Why Conv2D Upsampling?

1. **Reduced Checkerboard Artifacts**: Nearest neighbor upsampling followed by convolution reduces checkerboard patterns
2. **Better Control**: Separate upsampling and filtering operations provide more control
3. **Stable Training**: Less prone to training instabilities compared to transposed convolutions
4. **Flexible Design**: Easier to modify upsampling strategies

## Training Details

### Loss Functions
- **Generator Loss**: Binary Cross-Entropy (wants discriminator to classify fake as real)
- **Discriminator Loss**: Binary Cross-Entropy on real + fake samples

### Optimization
- **Optimizer**: Adam with β₁=0.5, β₂=0.999
- **Learning Rates**: 0.0002 for both generator and discriminator
- **Batch Size**: 64 (default, configurable)

### Training Schedule
- **Epochs**: 100 (default, configurable)
- **Sample Generation**: Every 5 epochs
- **Checkpoint Saving**: Every 10 epochs
- **Progress Monitoring**: TensorBoard integration

## Evaluation Metrics

### 1. Conditional Accuracy
Measures how well generated samples match their intended class labels using a trained classifier.

### 2. Inception Score (IS)
Evaluates the quality and diversity of generated samples:
- **Quality**: Generated samples should be classifiable
- **Diversity**: Generated samples should cover all classes

### 3. Class Diversity
Measures intra-class diversity by calculating average pairwise distances within each class.

### 4. Visual Quality
Subjective evaluation through sample grids and interpolations.

## Results and Performance

### Expected Results
- **Conditional Accuracy**: >85% (generated samples correctly classified)
- **Inception Score**: >2.0 for well-trained models
- **Visual Quality**: Clear, recognizable digits with proper class conditioning

### Training Time
- **CPU**: ~2-3 hours for 100 epochs
- **GPU (GTX 1080+)**: ~30-45 minutes for 100 epochs

## Advanced Features

### 1. Class Interpolation
Generate smooth transitions between different digit classes by interpolating in the label embedding space.

### 2. Conditional Generation
Generate specific digits on demand by providing class labels.

### 3. Comprehensive Monitoring
- Real-time loss tracking
- Sample visualization during training
- TensorBoard integration for detailed monitoring

### 4. Flexible Architecture
- Configurable feature map sizes
- Adjustable noise dimensions
- Modular design for easy experimentation

## Troubleshooting

### Common Issues

1. **Mode Collapse**: Generator produces similar samples
   - Solution: Adjust learning rates, try different optimizers

2. **Training Instability**: Loss oscillations
   - Solution: Lower learning rates, add noise to discriminator inputs

3. **Poor Conditional Control**: Generated samples don't match labels
   - Solution: Increase label embedding dimension, check data loading

4. **CUDA Out of Memory**: 
   - Solution: Reduce batch size, use gradient accumulation

### Performance Tips

1. **Use GPU**: Significantly faster training
2. **Monitor Early**: Check samples after 10-20 epochs
3. **Adjust Learning Rates**: Start with 0.0002, reduce if unstable
4. **Save Frequently**: Regular checkpoints prevent data loss

## Citations and References

This implementation is inspired by:
- **Conditional GANs**: Mirza & Osindero (2014)
- **DCGAN**: Radford et al. (2015)
- **Upsampling Techniques**: Odena et al. (2016)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
