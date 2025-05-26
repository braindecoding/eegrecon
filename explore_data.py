#!/usr/bin/env python3
"""
Quick data exploration script for MindBigData files
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from eeg_reconstruction import MindBigDataLoader

def explore_file(filepath, device_name):
    """Explore a single data file"""
    print(f"\n{'='*50}")
    print(f"EXPLORING {device_name} DATA: {filepath}")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    # Get file size
    file_size = os.path.getsize(filepath) / (1024*1024)  # MB
    print(f"File size: {file_size:.1f} MB")
    
    # Initialize loader
    loader = MindBigDataLoader(os.path.dirname(filepath))
    
    # Load first 1000 lines to get a sample
    print("Loading sample data...")
    try:
        with open(filepath, 'r') as f:
            lines = [f.readline().strip() for _ in range(1000) if f.readline()]
        
        # Parse sample lines
        records = []
        for line in lines[:100]:  # Parse first 100 lines
            if line.strip():
                record = loader.parse_line(line)
                if record:
                    records.append(record)
        
        if not records:
            print("No valid records found in sample")
            return
        
        print(f"Parsed {len(records)} sample records")
        
        # Analyze the data
        devices = set(r['device'] for r in records)
        channels = set(r['channel'] for r in records)
        codes = set(r['code'] for r in records)
        events = set(r['event'] for r in records)
        
        print(f"\nData Summary:")
        print(f"  Devices: {sorted(devices)}")
        print(f"  Channels: {sorted(channels)}")
        print(f"  Event codes: {sorted(codes)}")
        print(f"  Event IDs: {len(events)} unique events")
        
        # Signal analysis
        signal_lengths = [r['size'] for r in records]
        signal_values = []
        for r in records:
            if len(r['signal']) > 0:
                signal_values.extend(r['signal'])
        
        if signal_values:
            print(f"\nSignal Analysis:")
            print(f"  Signal lengths: {min(signal_lengths)} - {max(signal_lengths)} samples")
            print(f"  Signal range: {min(signal_values):.1f} - {max(signal_values):.1f}")
            print(f"  Signal mean: {np.mean(signal_values):.1f}")
            print(f"  Signal std: {np.std(signal_values):.1f}")
        
        # Channel-specific analysis
        print(f"\nChannel-specific analysis:")
        channel_data = {}
        for record in records:
            channel = record['channel']
            if channel not in channel_data:
                channel_data[channel] = []
            if len(record['signal']) > 0:
                channel_data[channel].extend(record['signal'])
        
        for channel, values in channel_data.items():
            if values:
                print(f"  {channel}: {len(values)} samples, mean={np.mean(values):.1f}, std={np.std(values):.1f}")
        
        # Event code analysis
        print(f"\nEvent code analysis:")
        code_counts = {}
        for record in records:
            code = record['code']
            code_counts[code] = code_counts.get(code, 0) + 1
        
        for code, count in sorted(code_counts.items()):
            print(f"  Code {code}: {count} records")
        
        # Try to visualize a sample signal
        if records and len(records[0]['signal']) > 10:
            try:
                plt.figure(figsize=(12, 6))
                
                # Plot signals from different channels
                channels_to_plot = list(channel_data.keys())[:4]  # Plot up to 4 channels
                
                for i, channel in enumerate(channels_to_plot):
                    if channel in channel_data and len(channel_data[channel]) >= 100:
                        plt.subplot(2, 2, i+1)
                        sample_signal = channel_data[channel][:100]  # First 100 samples
                        plt.plot(sample_signal)
                        plt.title(f'{device_name} - {channel}')
                        plt.xlabel('Sample')
                        plt.ylabel('Amplitude')
                        plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plot_filename = f'{device_name.lower()}_sample_signals.png'
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\nSample signals plot saved as: {plot_filename}")
                
            except Exception as e:
                print(f"Error creating plot: {e}")
        
    except Exception as e:
        print(f"Error exploring file: {e}")


def main():
    """Main exploration function"""
    print("EEG DATA EXPLORATION")
    print("="*60)
    
    # Define data files
    data_files = {
        'EPOC': 'data/EP1.01.txt',
        'MindWave': 'data/MW.txt', 
        'Muse': 'data/MU.txt',
        'Insight': 'data/IN.txt'
    }
    
    # Check data directory
    if not os.path.exists('data'):
        print("Error: 'data' directory not found!")
        return
    
    print("Available data files:")
    for device, filepath in data_files.items():
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024*1024)
            print(f"  ✓ {device}: {filepath} ({size:.1f} MB)")
        else:
            print(f"  ✗ {device}: {filepath} (not found)")
    
    # Explore each file
    for device, filepath in data_files.items():
        if os.path.exists(filepath):
            explore_file(filepath, device)
    
    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETED!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    for device in data_files.keys():
        plot_file = f"{device.lower()}_sample_signals.png"
        if os.path.exists(plot_file):
            print(f"  - {plot_file}")


if __name__ == "__main__":
    main()
