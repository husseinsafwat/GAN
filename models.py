import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator using Conv2D upsampling layers followed by kernel filters
    instead of Conv2DTranspose layers.
    """
    
    def __init__(self, noise_dim=100, num_classes=10, img_channels=1, feature_map_size=64):
        super(ConditionalGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.feature_map_size = feature_map_size
        
        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, noise_dim)
        
        # Initial dense layer to transform noise+label to feature maps
        self.initial_dense = nn.Sequential(
            nn.Linear(noise_dim * 2, feature_map_size * 8 * 7 * 7),
            nn.BatchNorm1d(feature_map_size * 8 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsampling layers with Conv2D + kernel filters
        # Layer 1: 7x7 -> 14x14
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(feature_map_size * 8, feature_map_size * 4, 
                              kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_map_size * 4)
        
        # Layer 2: 14x14 -> 28x28
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(feature_map_size * 4, feature_map_size * 2, 
                              kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_map_size * 2)
        
        # Layer 3: 28x28 -> 28x28 (refinement)
        self.conv3 = nn.Conv2d(feature_map_size * 2, feature_map_size, 
                              kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_map_size)
        
        # Layer 4: Final output layer
        self.conv4 = nn.Conv2d(feature_map_size, img_channels, 
                              kernel_size=3, stride=1, padding=1)
        
    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_embedding = self.label_embedding(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        
        # Transform to initial feature maps
        x = self.initial_dense(gen_input)
        x = x.view(-1, self.feature_map_size * 8, 7, 7)
        
        # Upsampling block 1
        x = self.upsample1(x)  # 7x7 -> 14x14
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Upsampling block 2
        x = self.upsample2(x)  # 14x14 -> 28x28
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Refinement block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Final output layer
        x = self.conv4(x)
        x = torch.tanh(x)
        
        return x


class ConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator that takes both image and label as input.
    """
    
    def __init__(self, img_channels=1, num_classes=10, feature_map_size=64):
        super(ConditionalDiscriminator, self).__init__()
        
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.feature_map_size = feature_map_size
        
        # Label embedding to match image dimensions
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(img_channels + 1, feature_map_size, 
                              kernel_size=4, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(feature_map_size, feature_map_size * 2, 
                              kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_map_size * 2)
        
        self.conv3 = nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 
                              kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_map_size * 4)
        
        # Final classification layer
        self.final = nn.Conv2d(feature_map_size * 4, 1, 
                              kernel_size=3, stride=1, padding=0)
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, img, labels):
        # Embed labels and reshape to match image dimensions
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(-1, 1, 28, 28)
        
        # Concatenate image and label
        x = torch.cat([img, label_embedding], dim=1)
        
        # Convolutional layers
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.dropout(x)
        
        # Final classification
        x = self.final(x)
        x = torch.sigmoid(x)
        
        return x.view(-1, 1)


def weights_init(m):
    """
    Initialize weights for the networks.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def create_models(noise_dim=100, num_classes=10, img_channels=1, feature_map_size=64):
    """
    Create and initialize generator and discriminator models.
    
    Args:
        noise_dim: Dimension of the input noise vector
        num_classes: Number of classes in the dataset
        img_channels: Number of channels in the images
        feature_map_size: Base number of feature maps
        
    Returns:
        generator, discriminator: Initialized models
    """
    generator = ConditionalGenerator(noise_dim, num_classes, img_channels, feature_map_size)
    discriminator = ConditionalDiscriminator(img_channels, num_classes, feature_map_size)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    return generator, discriminator