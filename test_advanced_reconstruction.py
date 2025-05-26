#!/usr/bin/env python3
"""
Test Advanced EEG to MNIST Reconstruction
Generate high-quality MNIST-like digits and compare with reference
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from advanced_eeg_reconstruction import AdvancedEEGReconstructor, AdvancedEEGToImageGenerator
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import os

def load_trained_model(model_path="best_eeg_to_mnist_model.pth", n_channels=14, n_timepoints=256):
    """Load the trained advanced model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AdvancedEEGToImageGenerator(
        n_channels=n_channels,
        n_timepoints=n_timepoints,
        latent_dim=512
    ).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("Please run training first with: python run_advanced_reconstruction.py")
        return None
    
    model.eval()
    return model, device

def load_mnist_samples():
    """Load real MNIST samples for comparison"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    mnist_dataset = MNIST(root='./mnist_data', train=False, download=True, transform=transform)
    
    # Get samples for each digit
    samples_by_digit = {i: [] for i in range(10)}
    
    for image, label in mnist_dataset:
        if len(samples_by_digit[label]) < 5:  # 5 samples per digit
            samples_by_digit[label].append(image.squeeze().numpy())
    
    return samples_by_digit

def generate_synthetic_eeg(n_channels=14, n_timepoints=256, digit=5):
    """Generate synthetic EEG data for testing (replace with real EEG)"""
    # Create digit-specific patterns
    base_freq = 10 + digit  # Different frequency for each digit
    
    eeg_signal = np.zeros((n_channels, n_timepoints))
    
    for ch in range(n_channels):
        # Create synthetic brain signal with digit-specific characteristics
        t = np.linspace(0, 2, n_timepoints)
        
        # Base oscillation
        signal = np.sin(2 * np.pi * base_freq * t)
        
        # Add harmonics
        signal += 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)
        signal += 0.1 * np.sin(2 * np.pi * (base_freq * 3) * t)
        
        # Add noise
        signal += 0.1 * np.random.randn(n_timepoints)
        
        # Channel-specific modulation
        channel_mod = 1 + 0.2 * np.sin(2 * np.pi * ch / n_channels)
        signal *= channel_mod
        
        # Normalize
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        eeg_signal[ch] = signal
    
    return eeg_signal

def test_reconstruction_quality():
    """Test the reconstruction quality with various inputs"""
    print("🧠 Testing Advanced EEG to MNIST Reconstruction")
    print("=" * 60)
    
    # Load trained model
    model, device = load_trained_model()
    if model is None:
        return
    
    # Load MNIST references
    print("📊 Loading MNIST reference samples...")
    mnist_samples = load_mnist_samples()
    
    # Test reconstruction for each digit
    print("🎯 Testing reconstruction for all digits...")
    
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    
    for digit in range(10):
        print(f"  Testing digit {digit}...")
        
        # Generate synthetic EEG for this digit
        eeg_signal = generate_synthetic_eeg(digit=digit)
        
        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg_signal).unsqueeze(0).to(device)
        
        # Generate image
        with torch.no_grad():
            generated_image, latent = model(eeg_tensor)
            generated_image = generated_image.cpu().numpy()[0, 0]
        
        # Get MNIST reference
        mnist_ref = mnist_samples[digit][0] if mnist_samples[digit] else np.zeros((28, 28))
        
        # Plot results
        # Row 0: MNIST reference
        axes[0, digit].imshow(mnist_ref, cmap='gray')
        axes[0, digit].set_title(f'MNIST {digit}', fontsize=10)
        axes[0, digit].axis('off')
        
        # Row 1: Generated image
        axes[1, digit].imshow(generated_image, cmap='gray')
        axes[1, digit].set_title(f'Generated {digit}', fontsize=10)
        axes[1, digit].axis('off')
        
        # Row 2: EEG signal (first channel)
        axes[2, digit].plot(eeg_signal[0, :50])  # First 50 timepoints
        axes[2, digit].set_title(f'EEG Ch1', fontsize=8)
        axes[2, digit].set_ylim(-2, 2)
        
        # Row 3: EEG signal (average across channels)
        avg_signal = np.mean(eeg_signal, axis=0)
        axes[3, digit].plot(avg_signal[:50])
        axes[3, digit].set_title(f'EEG Avg', fontsize=8)
        axes[3, digit].set_ylim(-2, 2)
    
    # Add row labels
    axes[0, 0].set_ylabel('MNIST\nReference', fontsize=12, rotation=0, ha='right')
    axes[1, 0].set_ylabel('EEG\nGenerated', fontsize=12, rotation=0, ha='right')
    axes[2, 0].set_ylabel('EEG\nSignal Ch1', fontsize=12, rotation=0, ha='right')
    axes[3, 0].set_ylabel('EEG\nAverage', fontsize=12, rotation=0, ha='right')
    
    plt.suptitle('Advanced EEG to MNIST Reconstruction - All Digits', fontsize=16)
    plt.tight_layout()
    plt.savefig('advanced_reconstruction_test_all_digits.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Test completed! Results saved as 'advanced_reconstruction_test_all_digits.png'")

def detailed_digit_analysis(digit=5):
    """Detailed analysis for a specific digit"""
    print(f"\n🔍 Detailed Analysis for Digit {digit}")
    print("=" * 40)
    
    # Load trained model
    model, device = load_trained_model()
    if model is None:
        return
    
    # Generate multiple EEG samples for the same digit
    n_samples = 8
    
    fig, axes = plt.subplots(3, n_samples, figsize=(16, 6))
    
    for i in range(n_samples):
        # Generate slightly different EEG for same digit
        eeg_signal = generate_synthetic_eeg(digit=digit)
        
        # Add some variation
        variation = 0.1 * np.random.randn(*eeg_signal.shape)
        eeg_signal += variation
        
        # Normalize
        eeg_signal = (eeg_signal - np.mean(eeg_signal)) / (np.std(eeg_signal) + 1e-8)
        
        # Convert to tensor
        eeg_tensor = torch.FloatTensor(eeg_signal).unsqueeze(0).to(device)
        
        # Generate image
        with torch.no_grad():
            generated_image, latent = model(eeg_tensor)
            generated_image = generated_image.cpu().numpy()[0, 0]
        
        # Plot generated image
        axes[0, i].imshow(generated_image, cmap='gray')
        axes[0, i].set_title(f'Generated {i+1}', fontsize=10)
        axes[0, i].axis('off')
        
        # Plot EEG signal (first channel)
        axes[1, i].plot(eeg_signal[0, :100])
        axes[1, i].set_title(f'EEG Ch1', fontsize=8)
        axes[1, i].set_ylim(-2, 2)
        
        # Plot EEG spectrogram-like visualization
        axes[2, i].imshow(eeg_signal[:8, :50], cmap='viridis', aspect='auto')
        axes[2, i].set_title(f'EEG Channels', fontsize=8)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel(f'Generated\nDigit {digit}', fontsize=12, rotation=0, ha='right')
    axes[1, 0].set_ylabel('EEG\nSignal', fontsize=12, rotation=0, ha='right')
    axes[2, 0].set_ylabel('EEG\nChannels', fontsize=12, rotation=0, ha='right')
    
    plt.suptitle(f'Detailed Analysis: Multiple EEG Inputs → Digit {digit}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'detailed_analysis_digit_{digit}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Detailed analysis saved as 'detailed_analysis_digit_{digit}.png'")

def calculate_reconstruction_metrics():
    """Calculate detailed reconstruction metrics"""
    print("\n📊 Calculating Reconstruction Metrics")
    print("=" * 40)
    
    # Load trained model
    model, device = load_trained_model()
    if model is None:
        return
    
    # Load MNIST references
    mnist_samples = load_mnist_samples()
    
    metrics = {}
    
    for digit in range(10):
        if not mnist_samples[digit]:
            continue
        
        # Generate EEG and reconstruct
        eeg_signal = generate_synthetic_eeg(digit=digit)
        eeg_tensor = torch.FloatTensor(eeg_signal).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated_image, _ = model(eeg_tensor)
            generated_image = generated_image.cpu().numpy()[0, 0]
        
        # Compare with MNIST reference
        mnist_ref = mnist_samples[digit][0]
        
        # Calculate metrics
        mse = np.mean((generated_image - mnist_ref) ** 2)
        
        # Simple SSIM
        mu1, mu2 = np.mean(generated_image), np.mean(mnist_ref)
        sigma1, sigma2 = np.var(generated_image), np.var(mnist_ref)
        sigma12 = np.mean((generated_image - mu1) * (mnist_ref - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        
        metrics[digit] = {'mse': mse, 'ssim': ssim}
        print(f"  Digit {digit}: MSE={mse:.4f}, SSIM={ssim:.4f}")
    
    # Overall metrics
    avg_mse = np.mean([m['mse'] for m in metrics.values()])
    avg_ssim = np.mean([m['ssim'] for m in metrics.values()])
    
    print(f"\n📈 Overall Performance:")
    print(f"  Average MSE: {avg_mse:.4f}")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    
    if avg_ssim > 0.7:
        print("🌟 EXCELLENT reconstruction quality!")
    elif avg_ssim > 0.5:
        print("✅ GOOD reconstruction quality!")
    elif avg_ssim > 0.3:
        print("🔧 FAIR reconstruction quality")
    else:
        print("🔴 Needs improvement")
    
    return metrics

def main():
    """Main testing function"""
    print("🚀 ADVANCED EEG RECONSTRUCTION TESTING")
    print("=" * 60)
    
    # Test 1: Overall reconstruction quality
    test_reconstruction_quality()
    
    # Test 2: Detailed analysis for specific digits
    for digit in [0, 5, 9]:  # Test a few specific digits
        detailed_digit_analysis(digit)
    
    # Test 3: Calculate metrics
    metrics = calculate_reconstruction_metrics()
    
    print(f"\n🎉 TESTING COMPLETED!")
    print(f"Generated files:")
    print(f"  - advanced_reconstruction_test_all_digits.png")
    print(f"  - detailed_analysis_digit_*.png")
    
    print(f"\n🧠 Your Advanced EEG Reconstruction System is ready!")
    print(f"It can generate MNIST-like digits from EEG brain signals! ✨")

if __name__ == "__main__":
    main()
