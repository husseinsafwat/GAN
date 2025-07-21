# Getting Started with Conditional GAN

This guide will help you get up and running with the Conditional GAN for MNIST digit generation in just a few minutes.

## 🚀 Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you don't have PyTorch installed, this may take a few minutes as it downloads the appropriate version for your system.

### Step 2: Run the Demo
```bash
python3 example.py
```

This will:
- ✅ Create the generator and discriminator models
- ✅ Download MNIST dataset (if not already present)
- ✅ Show real vs generated samples
- ✅ Demonstrate conditional generation
- ✅ Run a mini training session (5 steps)
- ✅ Save visualization images

**Expected Output:**
```
🚀 Conditional GAN Quick Demo
==================================================
📱 Device: cuda  # or cpu
🏗️  Creating models...
✅ Generator parameters: 6,593,873
✅ Discriminator parameters: 187,905
📊 Loading MNIST data...
✅ Dataset size: 60000
🖼️  Visualizing real samples...
✅ Real samples saved to 'real_samples.png'
...
🎉 Demo completed!
```

### Step 3: Check Generated Images
Look at the generated PNG files:
- `real_samples.png` - Real MNIST digits
- `untrained_samples.png` - Random noise (before training)
- `conditional_samples.png` - Digits 0-9 generation
- `mini_trained_samples.png` - After brief training

## 🏋️ Full Training (30-60 minutes)

### Basic Training
```bash
python3 train.py
```

**Default settings:**
- 100 epochs
- Batch size: 64
- Auto device detection (GPU if available)
- Saves checkpoints every 10 epochs
- Generates sample images every 5 epochs

### Advanced Training
```bash
python3 train.py \
  --epochs 200 \
  --batch_size 128 \
  --lr_g 0.0001 \
  --lr_d 0.0002 \
  --device cuda \
  --save_interval 20 \
  --sample_interval 10
```

**Training Output Structure:**
```
checkpoints/          # Model saves
├── checkpoint_epoch_0010.pth
├── checkpoint_epoch_0020.pth
└── final_model.pth

samples/              # Generated images during training
├── samples_epoch_0005.png
├── samples_epoch_0010.png
└── ...

logs/                 # TensorBoard logs
└── events.out.tfevents...
```

### Monitor Training with TensorBoard
```bash
tensorboard --logdir logs
```

Then open http://localhost:6006 in your browser.

## 🎨 Generate Custom Samples

### Generate Specific Digits
```bash
# Generate 64 samples of digit "7"
python3 inference.py \
  --checkpoint checkpoints/final_model.pth \
  --class_label 7 \
  --num_samples 64
```

### Generate All Digits
```bash
# Generate samples from all classes (0-9)
python3 inference.py \
  --checkpoint checkpoints/final_model.pth \
  --num_samples 80
```

### Class Interpolation
```bash
# Smooth transition from digit 0 to digit 9
python3 inference.py \
  --checkpoint checkpoints/final_model.pth \
  --interpolate \
  --interpolate_classes 0 9 \
  --interpolate_steps 10
```

## 📊 Evaluate Your Model

```bash
python3 evaluate.py \
  --checkpoint checkpoints/final_model.pth \
  --num_samples 10000
```

**Evaluation Metrics:**
- **Conditional Accuracy**: How well generated digits match their labels
- **Inception Score**: Quality and diversity of generated samples
- **Class Diversity**: Variation within each digit class
- **Visual Quality Assessment**

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python3 train.py --batch_size 32
```

**2. Slow Training**
```bash
# Check if GPU is being used
python3 -c "import torch; print(torch.cuda.is_available())"
```

**3. Poor Sample Quality**
```bash
# Train for more epochs with lower learning rate
python3 train.py --epochs 200 --lr_g 0.0001 --lr_d 0.0001
```

**4. Mode Collapse (all samples look similar)**
```bash
# Adjust learning rates
python3 train.py --lr_g 0.0001 --lr_d 0.0003
```

### Performance Tips

✅ **Use GPU**: 10-20x faster than CPU  
✅ **Monitor Early**: Check sample quality after 20-30 epochs  
✅ **Save Frequently**: Use `--save_interval 5` for experimentation  
✅ **Batch Size**: Start with 64, increase if you have more GPU memory  

## 📁 Project Structure

```
conditional-gan-mnist/
├── 📋 requirements.txt      # Dependencies
├── 📖 README.md            # Complete documentation  
├── 🏗️ ARCHITECTURE.md       # Technical details
├── 🚀 GETTING_STARTED.md   # This file
├── 💡 example.py           # Quick demo script
│
├── 🧠 models.py            # Generator & Discriminator
├── 📊 dataset.py           # Data loading & visualization
├── 🏋️ train.py             # Training script
├── 🎨 inference.py         # Sample generation
├── 📈 evaluate.py          # Model evaluation
│
└── 📁 Generated Directories:
    ├── data/               # MNIST dataset
    ├── checkpoints/        # Saved models
    ├── samples/            # Training samples
    ├── logs/              # TensorBoard logs
    ├── generated_samples/  # Inference outputs
    └── evaluation_results/ # Evaluation reports
```

## 🎯 Expected Results

### After 50 Epochs:
- Recognizable but blurry digits
- Some conditional control visible
- Inception Score: ~1.5-2.0

### After 100 Epochs:
- Clear, sharp digits
- Good conditional control
- Inception Score: ~2.0-2.5
- Conditional Accuracy: >80%

### After 200 Epochs:
- High-quality digits
- Excellent conditional control
- Inception Score: ~2.5+
- Conditional Accuracy: >90%

## 🆘 Need Help?

1. **Check the logs**: Look in `logs/` directory for training progress
2. **Visual inspection**: Check sample images in `samples/` directory  
3. **Read the docs**: See `README.md` and `ARCHITECTURE.md` for details
4. **Common patterns**: Mode collapse, training instability are normal initially

## 🎓 Next Steps

1. **Experiment with architecture**: Modify `models.py`
2. **Try different datasets**: Adapt `dataset.py` for other data
3. **Advanced techniques**: Add spectral normalization, progressive growing
4. **Evaluation**: Implement FID score, human evaluation

Happy generating! 🎨✨