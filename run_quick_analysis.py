#!/usr/bin/env python3
"""
Quick EEG reconstruction analysis with real MindBigData files (subset processing)
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from eeg_reconstruction import EEGExperimentPipeline, MindBigDataLoader

def load_subset_data(filepath, max_lines=5000):
    """Load a subset of data from a file for quick analysis"""
    print(f"Loading subset from {filepath} (max {max_lines} lines)...")

    loader = MindBigDataLoader(os.path.dirname(filepath))
    records = []

    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break

                if i % 1000 == 0:
                    print(f"  Processing line {i}...")

                line = line.strip()
                if line:
                    record = loader.parse_line(line)
                    if record:
                        records.append(record)

        print(f"  Loaded {len(records)} records")

        if not records:
            return np.array([]), [], []

        # Organize by trials
        trials = loader.organize_by_trials(records)
        print(f"  Organized into {len(trials)} trials")

        # Convert to arrays with consistent dimensions
        eeg_data = []
        labels = []
        trial_info = []

        # Find a common signal length (use median length)
        all_lengths = []
        for trial_key, trial_data in trials.items():
            if len(trial_data['channels']) > 0:
                channels = trial_data['channels']
                for sig in channels.values():
                    all_lengths.append(len(sig))

        if not all_lengths:
            return np.array([]), [], []

        # Use a reasonable signal length (median, but at least 100 samples)
        target_length = max(100, int(np.median(all_lengths)))
        target_length = min(target_length, 512)  # Cap at 512 for memory efficiency
        print(f"  Target signal length: {target_length}")

        for trial_key, trial_data in trials.items():
            if len(trial_data['channels']) > 0:
                # Get the signal data
                channels = trial_data['channels']

                # Create EEG matrix (channels x timepoints) with consistent length
                channel_names = sorted(channels.keys())
                eeg_matrix = []

                for ch in channel_names:
                    signal = channels[ch]

                    # Pad or truncate to target length
                    if len(signal) >= target_length:
                        # Truncate
                        processed_signal = signal[:target_length]
                    else:
                        # Pad with zeros
                        processed_signal = np.pad(signal, (0, target_length - len(signal)), 'constant')

                    eeg_matrix.append(processed_signal)

                if len(eeg_matrix) > 0:
                    eeg_matrix = np.array(eeg_matrix)
                    eeg_data.append(eeg_matrix)
                    labels.append(trial_data['code'])
                    trial_info.append(trial_data)

        if eeg_data:
            eeg_data = np.array(eeg_data)
            print(f"  Final shape: {eeg_data.shape}")
        else:
            eeg_data = np.array([])

        return eeg_data, labels, trial_info

    except Exception as e:
        print(f"  Error loading data: {e}")
        return np.array([]), [], []


def main():
    """Run quick EEG reconstruction analysis"""

    print("="*60)
    print("QUICK EEG RECONSTRUCTION ANALYSIS")
    print("="*60)

    # Define data files
    data_files = {
        'EP': 'data/EP1.01.txt',    # EPOC Emotiv
        'MW': 'data/MW.txt',        # MindWave
        'MU': 'data/MU.txt',        # Muse
        'IN': 'data/IN.txt'         # Insight
    }

    # Check files
    available_files = {}
    for device, filepath in data_files.items():
        if os.path.exists(filepath):
            available_files[device] = filepath
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"✓ {device}: {filepath} ({size:.1f} MB)")
        else:
            print(f"✗ {device}: {filepath} (not found)")

    if not available_files:
        print("No data files found!")
        return

    print(f"\nProcessing {len(available_files)} devices...")

    device_results = {}

    # Process each device
    for device, filepath in available_files.items():
        print(f"\n{'='*40}")
        print(f"PROCESSING {device} DEVICE")
        print(f"{'='*40}")

        # Load subset of data
        eeg_data, labels, trial_info = load_subset_data(filepath, max_lines=2000)

        if len(eeg_data) == 0:
            print(f"No valid data found for {device}")
            continue

        print(f"Loaded {len(eeg_data)} trials")
        print(f"EEG shape: {eeg_data.shape}")
        print(f"Unique labels: {len(set(labels))} ({sorted(set(labels))})")

        # Store results
        device_results[device] = {
            'n_samples': len(eeg_data),
            'n_channels': eeg_data.shape[1],
            'n_timepoints': eeg_data.shape[2],
            'n_unique_labels': len(set(labels)),
            'labels': labels,
            'eeg_data': eeg_data,
            'trial_info': trial_info,
            'accuracy': 0.0
        }

        # Quick classification test
        if len(eeg_data) >= 10 and len(set(labels)) >= 2:
            print(f"Running quick classification test...")

            try:
                # Initialize pipeline
                pipeline = EEGExperimentPipeline("data")

                # Use subset for classification
                max_samples = min(50, len(eeg_data))
                subset_eeg = eeg_data[:max_samples]
                subset_labels = labels[:max_samples]

                print(f"  Using {max_samples} samples for classification")

                # Run classification
                models = pipeline.compare_deep_learning_models(subset_eeg, subset_labels)

                # Get best accuracy
                best_accuracy = 0
                best_model = None
                for model_name, model_info in models.items():
                    if 'accuracy' in model_info:
                        if model_info['accuracy'] > best_accuracy:
                            best_accuracy = model_info['accuracy']
                            best_model = model_name

                device_results[device]['accuracy'] = best_accuracy
                device_results[device]['best_model'] = best_model
                device_results[device]['models'] = models

                print(f"  Best model: {best_model}")
                print(f"  Best accuracy: {best_accuracy:.3f}")

            except Exception as e:
                print(f"  Classification failed: {e}")
                device_results[device]['accuracy'] = 0.0

        # Signal analysis
        print(f"\nSignal Analysis:")
        signal_mean = np.mean(eeg_data)
        signal_std = np.std(eeg_data)
        signal_min = np.min(eeg_data)
        signal_max = np.max(eeg_data)

        print(f"  Signal range: {signal_min:.1f} to {signal_max:.1f}")
        print(f"  Signal mean: {signal_mean:.1f}")
        print(f"  Signal std: {signal_std:.1f}")

        device_results[device].update({
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'signal_min': signal_min,
            'signal_max': signal_max
        })

    # Summary analysis
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*60}")

    if len(device_results) > 0:
        print("\nDevice Comparison:")
        print(f"{'Device':<8} {'Samples':<8} {'Channels':<9} {'Timepoints':<11} {'Accuracy':<9}")
        print("-" * 55)

        for device, results in device_results.items():
            print(f"{device:<8} {results['n_samples']:<8} {results['n_channels']:<9} "
                  f"{results['n_timepoints']:<11} {results['accuracy']:<9.3f}")

        # Create visualization
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Plot 1: Number of samples
            devices = list(device_results.keys())
            samples = [device_results[d]['n_samples'] for d in devices]
            axes[0, 0].bar(devices, samples)
            axes[0, 0].set_title('Number of Samples per Device')
            axes[0, 0].set_ylabel('Samples')

            # Plot 2: Number of channels
            channels = [device_results[d]['n_channels'] for d in devices]
            axes[0, 1].bar(devices, channels)
            axes[0, 1].set_title('Number of Channels per Device')
            axes[0, 1].set_ylabel('Channels')

            # Plot 3: Classification accuracy
            accuracies = [device_results[d]['accuracy'] for d in devices]
            axes[1, 0].bar(devices, accuracies)
            axes[1, 0].set_title('Classification Accuracy per Device')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_ylim(0, 1)

            # Plot 4: Signal characteristics
            means = [device_results[d]['signal_mean'] for d in devices]
            stds = [device_results[d]['signal_std'] for d in devices]

            x = np.arange(len(devices))
            width = 0.35

            axes[1, 1].bar(x - width/2, means, width, label='Mean', alpha=0.7)
            axes[1, 1].bar(x + width/2, stds, width, label='Std', alpha=0.7)
            axes[1, 1].set_title('Signal Statistics per Device')
            axes[1, 1].set_ylabel('Amplitude')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(devices)
            axes[1, 1].legend()

            plt.tight_layout()
            plt.savefig('quick_analysis_results.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\nVisualization saved as: quick_analysis_results.png")

        except Exception as e:
            print(f"Visualization error: {e}")

        # Cross-device comparison
        if len(device_results) >= 2:
            print(f"\nCross-device insights:")

            # Find best performing device
            best_device = max(device_results.keys(),
                            key=lambda d: device_results[d]['accuracy'])
            print(f"  Best performing device: {best_device} "
                  f"(accuracy: {device_results[best_device]['accuracy']:.3f})")

            # Find device with most channels
            most_channels_device = max(device_results.keys(),
                                     key=lambda d: device_results[d]['n_channels'])
            print(f"  Most channels: {most_channels_device} "
                  f"({device_results[most_channels_device]['n_channels']} channels)")

            # Find device with most data
            most_data_device = max(device_results.keys(),
                                 key=lambda d: device_results[d]['n_samples'])
            print(f"  Most data: {most_data_device} "
                  f"({device_results[most_data_device]['n_samples']} samples)")

    # Save results
    try:
        import pickle
        with open('quick_analysis_results.pkl', 'wb') as f:
            pickle.dump(device_results, f)
        print(f"\nResults saved to: quick_analysis_results.pkl")
    except Exception as e:
        print(f"Error saving results: {e}")

    print(f"\n{'='*60}")
    print("QUICK ANALYSIS COMPLETED!")
    print(f"{'='*60}")

    print(f"\nProcessed {len(device_results)} devices successfully:")
    for device, results in device_results.items():
        print(f"  - {device}: {results['n_samples']} samples, "
              f"{results['n_channels']} channels, "
              f"accuracy: {results['accuracy']:.3f}")


if __name__ == "__main__":
    main()
