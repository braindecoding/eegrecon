#!/usr/bin/env python3
"""
Advanced EEG to MNIST Image Reconstruction with Enhanced Models
Optimized for high-quality digit reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from sklearn.metrics import accuracy_score
import os
from tqdm import tqdm

class AdvancedEEGToImageGenerator(nn.Module):
    """Enhanced EEG to Image Generator with attention and residual connections"""
    
    def __init__(self, n_channels=14, n_timepoints=256, latent_dim=512):
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.latent_dim = latent_dim
        
        # EEG Encoder with attention
        self.eeg_encoder = nn.Sequential(
            # Convolutional layers for spatial-temporal feature extraction
            nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Flatten()
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=128*32, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Feature compression to latent space
        self.feature_compressor = nn.Sequential(
            nn.Linear(128*32, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(latent_dim, latent_dim//2),
            nn.BatchNorm1d(latent_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Image decoder with residual connections
        self.image_decoder = nn.Sequential(
            nn.Linear(latent_dim//2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 784),  # 28x28
            nn.Sigmoid()
        )
        
    def forward(self, eeg_data):
        batch_size = eeg_data.size(0)
        
        # Add channel dimension if needed
        if len(eeg_data.shape) == 3:
            eeg_data = eeg_data.unsqueeze(1)  # (batch, 1, channels, timepoints)
        
        # EEG encoding
        features = self.eeg_encoder(eeg_data)  # (batch, 128*32)
        
        # Self-attention
        features_reshaped = features.unsqueeze(1)  # (batch, 1, features)
        attended_features, _ = self.attention(features_reshaped, features_reshaped, features_reshaped)
        attended_features = attended_features.squeeze(1)  # (batch, features)
        
        # Residual connection
        features = features + attended_features
        
        # Compress to latent space
        latent = self.feature_compressor(features)
        
        # Decode to image
        image = self.image_decoder(latent)
        image = image.view(batch_size, 1, 28, 28)
        
        return image, latent


class PerceptualLoss(nn.Module):
    """Perceptual loss for better visual quality"""
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def forward(self, generated, target):
        # Pixel-wise losses
        mse = self.mse_loss(generated, target)
        l1 = self.l1_loss(generated, target)
        
        # Gradient loss for edge preservation
        grad_loss = self.gradient_loss(generated, target)
        
        # Combined loss
        total_loss = mse + 0.1 * l1 + 0.05 * grad_loss
        return total_loss
    
    def gradient_loss(self, generated, target):
        # Compute gradients
        gen_grad_x = torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:])
        gen_grad_y = torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :])
        
        target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
        target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
        
        grad_loss = self.mse_loss(gen_grad_x, target_grad_x) + self.mse_loss(gen_grad_y, target_grad_y)
        return grad_loss


class EEGMNISTDataset(Dataset):
    """Dataset for EEG-MNIST pairs"""
    
    def __init__(self, eeg_data, labels, mnist_images):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)
        self.mnist_images = torch.FloatTensor(mnist_images)
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return self.eeg_data[idx], self.labels[idx], self.mnist_images[idx]


class AdvancedEEGReconstructor:
    """Advanced EEG to MNIST reconstruction system"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = PerceptualLoss()
        
    def create_model(self, n_channels, n_timepoints):
        """Create the advanced model"""
        self.model = AdvancedEEGToImageGenerator(
            n_channels=n_channels, 
            n_timepoints=n_timepoints,
            latent_dim=512
        ).to(self.device)
        
        # Advanced optimizer with scheduling
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
    def load_mnist_references(self):
        """Load MNIST dataset as reference images"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        mnist_dataset = MNIST(root='./mnist_data', train=True, download=True, transform=transform)
        
        # Organize by digit
        mnist_by_digit = {i: [] for i in range(10)}
        for image, label in mnist_dataset:
            mnist_by_digit[label].append(image.squeeze().numpy())
        
        # Convert to arrays
        for digit in range(10):
            mnist_by_digit[digit] = np.array(mnist_by_digit[digit])
        
        return mnist_by_digit
    
    def create_eeg_mnist_pairs(self, eeg_data, labels, mnist_refs, samples_per_digit=100):
        """Create EEG-MNIST pairs for training"""
        paired_eeg = []
        paired_images = []
        paired_labels = []
        
        print("Creating EEG-MNIST pairs...")
        
        for digit in range(10):
            # Get EEG samples for this digit
            digit_mask = np.array(labels) == digit
            digit_eeg = eeg_data[digit_mask]
            
            if len(digit_eeg) == 0:
                continue
                
            # Get MNIST references for this digit
            digit_mnist = mnist_refs[digit]
            
            # Create pairs
            n_pairs = min(samples_per_digit, len(digit_eeg), len(digit_mnist))
            
            for i in range(n_pairs):
                eeg_idx = i % len(digit_eeg)
                mnist_idx = np.random.randint(0, len(digit_mnist))
                
                paired_eeg.append(digit_eeg[eeg_idx])
                paired_images.append(digit_mnist[mnist_idx])
                paired_labels.append(digit)
        
        paired_eeg = np.array(paired_eeg)
        paired_images = np.array(paired_images)
        paired_labels = np.array(paired_labels)
        
        print(f"Created {len(paired_eeg)} EEG-MNIST pairs")
        return paired_eeg, paired_images, paired_labels
    
    def train_model(self, eeg_data, labels, epochs=100, batch_size=32, validation_split=0.2):
        """Train the advanced model"""
        print("Loading MNIST references...")
        mnist_refs = self.load_mnist_references()
        
        print("Creating EEG-MNIST pairs...")
        paired_eeg, paired_images, paired_labels = self.create_eeg_mnist_pairs(
            eeg_data, labels, mnist_refs, samples_per_digit=200
        )
        
        # Split data
        n_val = int(len(paired_eeg) * validation_split)
        indices = np.random.permutation(len(paired_eeg))
        
        train_eeg = paired_eeg[indices[n_val:]]
        train_images = paired_images[indices[n_val:]]
        train_labels = paired_labels[indices[n_val:]]
        
        val_eeg = paired_eeg[indices[:n_val]]
        val_images = paired_images[indices[:n_val]]
        val_labels = paired_labels[indices[:n_val]]
        
        # Create datasets
        train_dataset = EEGMNISTDataset(train_eeg, train_labels, train_images)
        val_dataset = EEGMNISTDataset(val_eeg, val_labels, val_images)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            
            for eeg_batch, label_batch, image_batch in train_pbar:
                eeg_batch = eeg_batch.to(self.device)
                image_batch = image_batch.to(self.device)
                
                self.optimizer.zero_grad()
                
                generated_images, latent = self.model(eeg_batch)
                loss = self.criterion(generated_images, image_batch.unsqueeze(1))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for eeg_batch, label_batch, image_batch in val_loader:
                    eeg_batch = eeg_batch.to(self.device)
                    image_batch = image_batch.to(self.device)
                    
                    generated_images, latent = self.model(eeg_batch)
                    loss = self.criterion(generated_images, image_batch.unsqueeze(1))
                    val_loss += loss.item()
            
            # Update learning rate
            self.scheduler.step()
            
            # Record losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'best_eeg_to_mnist_model.pth')
                print(f'New best model saved! Val Loss: {avg_val_loss:.4f}')
            
            # Generate sample images every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.generate_sample_images(val_eeg[:8], val_images[:8], val_labels[:8], epoch+1)
        
        return train_losses, val_losses
    
    def generate_sample_images(self, eeg_samples, target_images, labels, epoch):
        """Generate and save sample images"""
        self.model.eval()
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_samples).to(self.device)
            generated_images, _ = self.model(eeg_tensor)
            generated_images = generated_images.cpu().numpy()
        
        fig, axes = plt.subplots(3, 8, figsize=(16, 6))
        
        for i in range(min(8, len(eeg_samples))):
            # Original MNIST
            axes[0, i].imshow(target_images[i], cmap='gray')
            axes[0, i].set_title(f'Target: {labels[i]}')
            axes[0, i].axis('off')
            
            # Generated image
            axes[1, i].imshow(generated_images[i, 0], cmap='gray')
            axes[1, i].set_title(f'Generated: {labels[i]}')
            axes[1, i].axis('off')
            
            # EEG signal (first channel)
            axes[2, i].plot(eeg_samples[i, 0, :100])  # First 100 timepoints
            axes[2, i].set_title(f'EEG Signal')
            axes[2, i].set_ylim(-1, 1)
        
        plt.suptitle(f'EEG to MNIST Reconstruction - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'reconstruction_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'Sample images saved: reconstruction_epoch_{epoch}.png')
    
    def evaluate_model(self, eeg_data, labels):
        """Evaluate the trained model"""
        print("Evaluating model...")
        mnist_refs = self.load_mnist_references()
        
        # Create test pairs
        test_eeg, test_images, test_labels = self.create_eeg_mnist_pairs(
            eeg_data, labels, mnist_refs, samples_per_digit=50
        )
        
        self.model.eval()
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(test_eeg).to(self.device)
            generated_images, _ = self.model(eeg_tensor)
            generated_images = generated_images.cpu().numpy()
        
        # Calculate metrics
        mse_scores = []
        ssim_scores = []
        
        for i in range(len(test_images)):
            target = test_images[i]
            generated = generated_images[i, 0]
            
            # MSE
            mse = np.mean((target - generated) ** 2)
            mse_scores.append(mse)
            
            # Simple SSIM approximation
            ssim = self.calculate_ssim(target, generated)
            ssim_scores.append(ssim)
        
        avg_mse = np.mean(mse_scores)
        avg_ssim = np.mean(ssim_scores)
        
        print(f"Evaluation Results:")
        print(f"  Average MSE: {avg_mse:.4f}")
        print(f"  Average SSIM: {avg_ssim:.4f}")
        
        # Generate final comparison
        self.generate_final_comparison(test_eeg[:10], test_images[:10], test_labels[:10])
        
        return avg_mse, avg_ssim
    
    def calculate_ssim(self, img1, img2):
        """Simple SSIM calculation"""
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    def generate_final_comparison(self, eeg_samples, target_images, labels):
        """Generate final comparison visualization"""
        self.model.eval()
        with torch.no_grad():
            eeg_tensor = torch.FloatTensor(eeg_samples).to(self.device)
            generated_images, _ = self.model(eeg_tensor)
            generated_images = generated_images.cpu().numpy()
        
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        
        for i in range(10):
            # Target MNIST
            axes[0, i].imshow(target_images[i], cmap='gray')
            axes[0, i].set_title(f'Target: {labels[i]}', fontsize=12)
            axes[0, i].axis('off')
            
            # Generated image
            axes[1, i].imshow(generated_images[i, 0], cmap='gray')
            axes[1, i].set_title(f'Generated: {labels[i]}', fontsize=12)
            axes[1, i].axis('off')
        
        axes[0, 0].set_ylabel('Target MNIST', fontsize=14)
        axes[1, 0].set_ylabel('EEG Generated', fontsize=14)
        
        plt.suptitle('Final EEG to MNIST Reconstruction Results', fontsize=16)
        plt.tight_layout()
        plt.savefig('final_reconstruction_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Final comparison saved: final_reconstruction_comparison.png")


def main():
    """Main function to run advanced EEG reconstruction"""
    print("🧠 Advanced EEG to MNIST Reconstruction")
    print("=" * 50)
    
    # Initialize reconstructor
    reconstructor = AdvancedEEGReconstructor()
    
    # This would be replaced with your real EEG data loading
    print("Note: Replace this section with your real EEG data loading")
    print("For demo, we'll create synthetic data...")
    
    # Demo with synthetic data (replace with your real data)
    n_samples = 1000
    n_channels = 14
    n_timepoints = 256
    
    # Synthetic EEG data
    eeg_data = np.random.randn(n_samples, n_channels, n_timepoints) * 0.1
    labels = np.random.randint(0, 10, n_samples)
    
    # Create model
    reconstructor.create_model(n_channels, n_timepoints)
    
    # Train model
    print("\nStarting advanced training...")
    train_losses, val_losses = reconstructor.train_model(
        eeg_data, labels, epochs=50, batch_size=16
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    mse, ssim = reconstructor.evaluate_model(eeg_data, labels)
    
    print(f"\n🎉 Training completed!")
    print(f"Final MSE: {mse:.4f}")
    print(f"Final SSIM: {ssim:.4f}")
    
    return reconstructor


if __name__ == "__main__":
    reconstructor = main()
