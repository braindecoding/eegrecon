#!/usr/bin/env python3
"""
Script to run EEG reconstruction analysis with real MindBigData files
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from eeg_reconstruction import EEGExperimentPipeline

def main():
    """Run the complete EEG reconstruction analysis with real data"""

    print("="*60)
    print("EEG RECONSTRUCTION ANALYSIS WITH REAL DATA")
    print("="*60)

    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found!")
        return

    # Define the data files
    data_files = {
        'EP': 'data/EP1.01.txt',    # EPOC Emotiv (14 channels)
        'MW': 'data/MW.txt',        # MindWave (1 channel)
        'MU': 'data/MU.txt',        # Muse (4 channels)
        'IN': 'data/IN.txt'         # Insight (5 channels)
    }

    # Check if all files exist
    missing_files = []
    for device, filepath in data_files.items():
        if not os.path.exists(filepath):
            missing_files.append(f"{device}: {filepath}")

    if missing_files:
        print("Error: Missing data files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return

    print("Found all data files:")
    for device, filepath in data_files.items():
        file_size = os.path.getsize(filepath) / (1024*1024)  # MB
        print(f"  - {device}: {filepath} ({file_size:.1f} MB)")

    print("\n" + "="*60)
    print("INITIALIZING EXPERIMENT PIPELINE")
    print("="*60)

    # Initialize the experiment pipeline
    pipeline = EEGExperimentPipeline(data_dir)

    print("\n1. LOADING AND ANALYZING DATA BY DEVICE")
    print("-" * 40)

    device_results = {}

    for device, filepath in data_files.items():
        print(f"\nProcessing {device} device...")

        try:
            # Load data for this device
            eeg_data, labels, trial_info = pipeline.data_loader.load_by_device([filepath])

            if len(eeg_data) == 0:
                print(f"  Warning: No data found for device {device}")
                continue

            print(f"  Loaded: {len(eeg_data)} samples")
            print(f"  Shape: {eeg_data.shape}")
            print(f"  Unique labels: {len(set(labels))}")

            # Store basic statistics
            device_results[device] = {
                'n_samples': len(eeg_data),
                'n_channels': eeg_data.shape[1],
                'n_timepoints': eeg_data.shape[2],
                'n_unique_labels': len(set(labels)),
                'labels': labels,
                'eeg_data': eeg_data,
                'trial_info': trial_info
            }

            # Run classification analysis if we have enough data
            if len(eeg_data) >= 10 and len(set(labels)) >= 2:
                print(f"  Running classification analysis...")

                # Use a subset for faster processing
                max_samples = min(100, len(eeg_data))
                subset_eeg = eeg_data[:max_samples]
                subset_labels = labels[:max_samples]

                try:
                    models = pipeline.compare_deep_learning_models(subset_eeg, subset_labels)

                    # Get best accuracy
                    best_accuracy = 0
                    for model_name, model_info in models.items():
                        if 'accuracy' in model_info:
                            best_accuracy = max(best_accuracy, model_info['accuracy'])

                    device_results[device]['accuracy'] = best_accuracy
                    device_results[device]['models'] = models

                    print(f"  Best accuracy: {best_accuracy:.3f}")

                except Exception as e:
                    print(f"  Classification failed: {e}")
                    device_results[device]['accuracy'] = 0.0
            else:
                print(f"  Skipping classification (insufficient data)")
                device_results[device]['accuracy'] = 0.0

        except Exception as e:
            print(f"  Error processing {device}: {e}")
            continue

    print("\n2. DEVICE COMPARISON ANALYSIS")
    print("-" * 40)

    if len(device_results) >= 2:
        # Compare devices
        print("\nDevice Comparison Summary:")
        print(f"{'Device':<8} {'Samples':<8} {'Channels':<9} {'Accuracy':<9}")
        print("-" * 40)

        for device, results in device_results.items():
            print(f"{device:<8} {results['n_samples']:<8} {results['n_channels']:<9} {results.get('accuracy', 0):<9.3f}")

        # Visualize device comparison
        try:
            pipeline.visualization_tools.plot_device_comparison(device_results)
            plt.savefig('device_comparison.png', dpi=150, bbox_inches='tight')
            print("\nDevice comparison plot saved as 'device_comparison.png'")
        except Exception as e:
            print(f"Visualization error: {e}")

    print("\n3. CROSS-DEVICE VALIDATION")
    print("-" * 40)

    # Cross-device validation (if we have multiple devices with sufficient data)
    devices_with_data = [d for d, r in device_results.items()
                        if r['n_samples'] >= 20 and r.get('accuracy', 0) > 0]

    if len(devices_with_data) >= 2:
        print(f"Running cross-device validation between: {devices_with_data}")

        cross_device_results = {}

        for source_device in devices_with_data:
            for target_device in devices_with_data:
                if source_device != target_device:
                    try:
                        print(f"  Training on {source_device}, testing on {target_device}...")

                        # Get source data
                        source_data = device_results[source_device]['eeg_data'][:50]  # Limit for speed
                        source_labels = device_results[source_device]['labels'][:50]

                        # Get target data
                        target_data = device_results[target_device]['eeg_data'][:50]
                        target_labels = device_results[target_device]['labels'][:50]

                        # Run cross-device validation
                        accuracy = pipeline.cross_device_validation(
                            source_data, source_labels, target_data, target_labels
                        )

                        cross_device_results[f"{source_device}_to_{target_device}"] = accuracy
                        print(f"    Accuracy: {accuracy:.3f}")

                    except Exception as e:
                        print(f"    Error: {e}")
                        continue

        if cross_device_results:
            print("\nCross-device validation results:")
            for pair, accuracy in cross_device_results.items():
                print(f"  {pair}: {accuracy:.3f}")

            # Store results
            pipeline.results['cross_device'] = cross_device_results

    print("\n4. IMAGE RECONSTRUCTION ANALYSIS")
    print("-" * 40)

    # Try image reconstruction with the device that has the most data
    best_device = None
    max_samples = 0

    for device, results in device_results.items():
        if results['n_samples'] > max_samples:
            max_samples = results['n_samples']
            best_device = device

    if best_device and max_samples >= 20:
        print(f"Running image reconstruction with {best_device} device...")

        try:
            # Get data for image reconstruction
            eeg_data = device_results[best_device]['eeg_data'][:50]  # Limit for speed
            labels = device_results[best_device]['labels'][:50]

            # Run image reconstruction
            reconstruction_results = pipeline.image_reconstruction_analysis(eeg_data, labels)

            print(f"Image reconstruction completed!")
            print(f"  MSE: {reconstruction_results.get('mse', 'N/A')}")
            print(f"  SSIM: {reconstruction_results.get('ssim', 'N/A')}")

        except Exception as e:
            print(f"Image reconstruction failed: {e}")

    print("\n5. GENERATING COMPREHENSIVE REPORT")
    print("-" * 40)

    # Generate final report
    pipeline.results['device_analysis'] = device_results
    pipeline.generate_comprehensive_report()

    # Save results
    try:
        import pickle
        with open('eeg_analysis_results.pkl', 'wb') as f:
            pickle.dump(pipeline.results, f)
        print("\nResults saved to 'eeg_analysis_results.pkl'")
    except Exception as e:
        print(f"Error saving results: {e}")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETED!")
    print("="*60)
    print("\nFiles generated:")
    print("  - device_comparison.png (if successful)")
    print("  - eeg_analysis_results.pkl")
    print("  - Various plots from the analysis")

    print(f"\nProcessed {len(device_results)} devices:")
    for device, results in device_results.items():
        print(f"  - {device}: {results['n_samples']} samples, {results['n_channels']} channels")


if __name__ == "__main__":
    main()
