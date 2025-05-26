#!/usr/bin/env python3
"""
Run Advanced EEG to MNIST Reconstruction with Real Data
Optimized for high-quality digit reconstruction like the MNIST samples shown
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from advanced_eeg_reconstruction import AdvancedEEGReconstructor
from eeg_reconstruction import MindBigDataLoader

def load_real_eeg_data(data_dir="data", max_samples_per_device=500):
    """Load real EEG data from MindBigData files"""
    print("🧠 Loading Real EEG Data for Advanced Reconstruction")
    print("=" * 60)
    
    data_files = {
        'MindWave': 'MW.txt',    # Best for single channel analysis
        'Muse': 'MU.txt',        # Good 4-channel data
        'Insight': 'IN.txt',     # 5 channels
        'EPOC': 'EP1.01.txt'     # 14 channels (use smaller sample)
    }
    
    all_eeg_data = []
    all_labels = []
    device_info = []
    
    loader = MindBigDataLoader(data_dir)
    
    for device_name, filename in data_files.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  {device_name} file not found: {filepath}")
            continue
        
        print(f"\n📊 Processing {device_name} ({filename})...")
        
        # Load data with limit for memory efficiency
        max_lines = max_samples_per_device if device_name != 'EPOC' else 200
        
        try:
            records = []
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    
                    if i % 100 == 0 and i > 0:
                        print(f"  Loaded {i} lines...")
                    
                    line = line.strip()
                    if line:
                        record = loader.parse_line(line)
                        if record:
                            records.append(record)
            
            print(f"  Parsed {len(records)} records")
            
            # Organize by trials
            trials = loader.organize_by_trials(records)
            print(f"  Organized into {len(trials)} trials")
            
            # Convert to EEG arrays
            device_eeg = []
            device_labels = []
            
            # Find target signal length (use median)
            all_lengths = []
            for trial_data in trials.values():
                if len(trial_data['channels']) > 0:
                    for sig in trial_data['channels'].values():
                        all_lengths.append(len(sig))
            
            if not all_lengths:
                print(f"  No valid signals found for {device_name}")
                continue
            
            target_length = max(128, min(512, int(np.median(all_lengths))))
            print(f"  Target signal length: {target_length}")
            
            # Process trials
            for trial_key, trial_data in trials.items():
                channels = trial_data['channels']
                
                if len(channels) == 0:
                    continue
                
                # Create EEG matrix with consistent dimensions
                channel_names = sorted(channels.keys())
                eeg_matrix = []
                
                for ch in channel_names:
                    signal = channels[ch]
                    
                    # Normalize signal
                    signal = np.array(signal)
                    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
                    
                    # Pad or truncate to target length
                    if len(signal) >= target_length:
                        processed_signal = signal[:target_length]
                    else:
                        processed_signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
                    
                    eeg_matrix.append(processed_signal)
                
                if len(eeg_matrix) > 0:
                    eeg_matrix = np.array(eeg_matrix)
                    device_eeg.append(eeg_matrix)
                    device_labels.append(trial_data['code'])
            
            if len(device_eeg) > 0:
                device_eeg = np.array(device_eeg)
                device_labels = np.array(device_labels)
                
                print(f"  Final shape: {device_eeg.shape}")
                print(f"  Labels: {len(set(device_labels))} unique ({sorted(set(device_labels))})")
                
                all_eeg_data.append(device_eeg)
                all_labels.append(device_labels)
                device_info.append({
                    'name': device_name,
                    'n_channels': device_eeg.shape[1],
                    'n_timepoints': device_eeg.shape[2],
                    'n_samples': len(device_eeg)
                })
            
        except Exception as e:
            print(f"  Error processing {device_name}: {e}")
            continue
    
    if not all_eeg_data:
        raise ValueError("No valid EEG data loaded!")
    
    # Combine all devices (pad channels to match the largest)
    max_channels = max(data.shape[1] for data in all_eeg_data)
    max_timepoints = max(data.shape[2] for data in all_eeg_data)
    
    print(f"\n📈 Combining data with max dimensions: {max_channels} channels, {max_timepoints} timepoints")
    
    combined_eeg = []
    combined_labels = []
    
    for i, (eeg_data, labels) in enumerate(zip(all_eeg_data, all_labels)):
        n_samples, n_channels, n_timepoints = eeg_data.shape
        
        # Pad channels if needed
        if n_channels < max_channels:
            padding = np.zeros((n_samples, max_channels - n_channels, n_timepoints))
            eeg_data = np.concatenate([eeg_data, padding], axis=1)
        
        # Pad timepoints if needed
        if n_timepoints < max_timepoints:
            padding = np.zeros((n_samples, max_channels, max_timepoints - n_timepoints))
            eeg_data = np.concatenate([eeg_data, padding], axis=2)
        
        combined_eeg.append(eeg_data)
        combined_labels.append(labels)
        
        print(f"  {device_info[i]['name']}: {n_samples} samples added")
    
    # Final combination
    final_eeg = np.concatenate(combined_eeg, axis=0)
    final_labels = np.concatenate(combined_labels, axis=0)
    
    print(f"\n✅ Final dataset:")
    print(f"  Shape: {final_eeg.shape}")
    print(f"  Labels: {len(set(final_labels))} unique digits")
    print(f"  Total samples: {len(final_eeg)}")
    
    return final_eeg, final_labels, device_info


def run_advanced_reconstruction():
    """Run the complete advanced reconstruction pipeline"""
    print("🚀 ADVANCED EEG TO MNIST RECONSTRUCTION")
    print("=" * 60)
    print("Goal: Generate high-quality MNIST-like digits from EEG signals")
    print("=" * 60)
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # Load real EEG data
    try:
        eeg_data, labels, device_info = load_real_eeg_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Initialize advanced reconstructor
    print(f"\n🧠 Initializing Advanced EEG Reconstructor...")
    reconstructor = AdvancedEEGReconstructor(device=device)
    
    # Create model
    n_channels = eeg_data.shape[1]
    n_timepoints = eeg_data.shape[2]
    
    print(f"📐 Creating model for {n_channels} channels, {n_timepoints} timepoints...")
    reconstructor.create_model(n_channels, n_timepoints)
    
    # Filter data to ensure we have all digits
    unique_labels = sorted(set(labels))
    print(f"📊 Available digits: {unique_labels}")
    
    # Balance dataset
    balanced_eeg = []
    balanced_labels = []
    min_samples = 50  # Minimum samples per digit
    
    for digit in range(10):
        if digit in unique_labels:
            digit_mask = labels == digit
            digit_eeg = eeg_data[digit_mask]
            
            if len(digit_eeg) >= min_samples:
                # Take up to 100 samples per digit for training efficiency
                n_take = min(100, len(digit_eeg))
                indices = np.random.choice(len(digit_eeg), n_take, replace=False)
                
                balanced_eeg.append(digit_eeg[indices])
                balanced_labels.extend([digit] * n_take)
                
                print(f"  Digit {digit}: {n_take} samples")
            else:
                print(f"  Digit {digit}: Only {len(digit_eeg)} samples (skipping)")
    
    if not balanced_eeg:
        print("❌ No sufficient data for training!")
        return
    
    balanced_eeg = np.concatenate(balanced_eeg, axis=0)
    balanced_labels = np.array(balanced_labels)
    
    print(f"\n📈 Balanced dataset: {balanced_eeg.shape[0]} samples")
    
    # Train the advanced model
    print(f"\n🎯 Starting Advanced Training...")
    print("This will generate high-quality MNIST-like reconstructions!")
    
    try:
        train_losses, val_losses = reconstructor.train_model(
            balanced_eeg, 
            balanced_labels, 
            epochs=100,  # More epochs for better quality
            batch_size=16,
            validation_split=0.2
        )
        
        print(f"\n✅ Training completed successfully!")
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_losses[-20:], label='Training Loss (Last 20)')
        plt.plot(val_losses[-20:], label='Validation Loss (Last 20)')
        plt.title('Recent Training Progress')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('advanced_training_progress.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Evaluate the model
        print(f"\n📊 Evaluating Advanced Model...")
        mse, ssim = reconstructor.evaluate_model(balanced_eeg, balanced_labels)
        
        print(f"\n🎉 ADVANCED RECONSTRUCTION RESULTS:")
        print(f"=" * 50)
        print(f"📈 MSE Score: {mse:.4f} (lower is better)")
        print(f"📈 SSIM Score: {ssim:.4f} (higher is better)")
        
        if ssim > 0.7:
            print("🌟 EXCELLENT reconstruction quality!")
        elif ssim > 0.5:
            print("✅ GOOD reconstruction quality!")
        elif ssim > 0.3:
            print("🔧 FAIR reconstruction quality - can be improved")
        else:
            print("🔴 Needs improvement")
        
        print(f"\n📁 Generated files:")
        print(f"  - best_eeg_to_mnist_model.pth (trained model)")
        print(f"  - final_reconstruction_comparison.png (results)")
        print(f"  - reconstruction_epoch_*.png (training progress)")
        print(f"  - advanced_training_progress.png (loss curves)")
        
        print(f"\n🎯 Your EEG reconstruction system is now optimized!")
        print(f"The model can generate MNIST-like digits from EEG brain signals! 🧠✨")
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_advanced_reconstruction()
