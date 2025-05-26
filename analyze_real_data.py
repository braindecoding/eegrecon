#!/usr/bin/env python3
"""
Simple analysis of real EEG data without deep learning training
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from eeg_reconstruction import MindBigDataLoader

def load_and_analyze_device(filepath, device_name, max_lines=1000):
    """Load and analyze data from a single device"""
    print(f"\n{'='*50}")
    print(f"ANALYZING {device_name} DEVICE")
    print(f"{'='*50}")
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    # Get file info
    file_size = os.path.getsize(filepath) / (1024*1024)
    print(f"File: {filepath} ({file_size:.1f} MB)")
    
    # Load data
    loader = MindBigDataLoader(os.path.dirname(filepath))
    records = []
    
    print(f"Loading first {max_lines} lines...")
    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                
                if i % 500 == 0 and i > 0:
                    print(f"  Processed {i} lines...")
                
                line = line.strip()
                if line:
                    record = loader.parse_line(line)
                    if record:
                        records.append(record)
        
        print(f"Successfully parsed {len(records)} records")
        
        if not records:
            print("No valid records found")
            return None
        
        # Organize by trials
        trials = loader.organize_by_trials(records)
        print(f"Organized into {len(trials)} trials")
        
        # Basic statistics
        devices = set(r['device'] for r in records)
        channels = set(r['channel'] for r in records)
        codes = set(r['code'] for r in records)
        events = set(r['event'] for r in records)
        
        print(f"\nBasic Statistics:")
        print(f"  Devices: {sorted(devices)}")
        print(f"  Channels: {sorted(channels)}")
        print(f"  Event codes: {sorted(codes)}")
        print(f"  Unique events: {len(events)}")
        
        # Signal analysis
        all_signals = []
        signal_lengths = []
        
        for record in records:
            if len(record['signal']) > 0:
                all_signals.extend(record['signal'])
                signal_lengths.append(len(record['signal']))
        
        if all_signals:
            print(f"\nSignal Analysis:")
            print(f"  Total signal samples: {len(all_signals)}")
            print(f"  Signal length range: {min(signal_lengths)} - {max(signal_lengths)}")
            print(f"  Signal value range: {min(all_signals):.1f} - {max(all_signals):.1f}")
            print(f"  Signal mean: {np.mean(all_signals):.1f}")
            print(f"  Signal std: {np.std(all_signals):.1f}")
        
        # Channel analysis
        print(f"\nChannel Analysis:")
        channel_stats = {}
        for record in records:
            channel = record['channel']
            if channel not in channel_stats:
                channel_stats[channel] = []
            if len(record['signal']) > 0:
                channel_stats[channel].extend(record['signal'])
        
        for channel, values in channel_stats.items():
            if values:
                print(f"  {channel}: {len(values)} samples, "
                      f"mean={np.mean(values):.1f}, std={np.std(values):.1f}")
        
        # Event code analysis
        print(f"\nEvent Code Analysis:")
        code_counts = {}
        for record in records:
            code = record['code']
            code_counts[code] = code_counts.get(code, 0) + 1
        
        for code in sorted(code_counts.keys()):
            print(f"  Code {code}: {code_counts[code]} records")
        
        # Try to create EEG matrix for a few trials
        print(f"\nTrial Analysis:")
        valid_trials = 0
        trial_shapes = []
        
        for trial_key, trial_data in list(trials.items())[:10]:  # Check first 10 trials
            channels = trial_data['channels']
            if len(channels) > 0:
                # Get signal lengths for this trial
                lengths = [len(sig) for sig in channels.values()]
                min_length = min(lengths)
                max_length = max(lengths)
                
                if min_length > 50:  # Only consider trials with reasonable signal length
                    valid_trials += 1
                    trial_shapes.append((len(channels), min_length))
                    
                    if valid_trials <= 3:  # Show details for first 3 valid trials
                        print(f"  Trial {trial_key}: {len(channels)} channels, "
                              f"length {min_length}-{max_length}")
        
        print(f"  Valid trials (length > 50): {valid_trials}")
        
        if trial_shapes:
            channels_list = [shape[0] for shape in trial_shapes]
            lengths_list = [shape[1] for shape in trial_shapes]
            print(f"  Channel count range: {min(channels_list)} - {max(channels_list)}")
            print(f"  Signal length range: {min(lengths_list)} - {max(lengths_list)}")
        
        # Create visualization
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Plot 1: Signal distribution
            if all_signals:
                axes[0, 0].hist(all_signals[:10000], bins=50, alpha=0.7)  # Sample for speed
                axes[0, 0].set_title(f'{device_name} - Signal Distribution')
                axes[0, 0].set_xlabel('Amplitude')
                axes[0, 0].set_ylabel('Frequency')
            
            # Plot 2: Event code distribution
            codes = list(code_counts.keys())
            counts = list(code_counts.values())
            axes[0, 1].bar(codes, counts)
            axes[0, 1].set_title(f'{device_name} - Event Code Distribution')
            axes[0, 1].set_xlabel('Event Code')
            axes[0, 1].set_ylabel('Count')
            
            # Plot 3: Channel statistics
            if channel_stats:
                channels = list(channel_stats.keys())
                means = [np.mean(channel_stats[ch]) for ch in channels]
                axes[1, 0].bar(channels, means)
                axes[1, 0].set_title(f'{device_name} - Channel Means')
                axes[1, 0].set_xlabel('Channel')
                axes[1, 0].set_ylabel('Mean Amplitude')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Plot 4: Sample signal
            if channel_stats:
                first_channel = list(channel_stats.keys())[0]
                sample_signal = channel_stats[first_channel][:500]  # First 500 samples
                axes[1, 1].plot(sample_signal)
                axes[1, 1].set_title(f'{device_name} - Sample Signal ({first_channel})')
                axes[1, 1].set_xlabel('Sample')
                axes[1, 1].set_ylabel('Amplitude')
            
            plt.tight_layout()
            plot_filename = f'{device_name.lower()}_analysis.png'
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nVisualization saved: {plot_filename}")
            
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Return summary
        return {
            'device': device_name,
            'file_size_mb': file_size,
            'n_records': len(records),
            'n_trials': len(trials),
            'n_channels': len(channels),
            'channels': sorted(channels),
            'event_codes': sorted(codes),
            'n_events': len(events),
            'signal_stats': {
                'count': len(all_signals),
                'mean': np.mean(all_signals) if all_signals else 0,
                'std': np.std(all_signals) if all_signals else 0,
                'min': min(all_signals) if all_signals else 0,
                'max': max(all_signals) if all_signals else 0
            },
            'valid_trials': valid_trials,
            'channel_stats': {ch: {'mean': np.mean(vals), 'std': np.std(vals)} 
                            for ch, vals in channel_stats.items() if vals}
        }
        
    except Exception as e:
        print(f"Error analyzing {device_name}: {e}")
        return None


def main():
    """Main analysis function"""
    print("="*60)
    print("REAL EEG DATA ANALYSIS")
    print("="*60)
    
    # Define data files
    data_files = {
        'EPOC': 'data/EP1.01.txt',
        'MindWave': 'data/MW.txt',
        'Muse': 'data/MU.txt',
        'Insight': 'data/IN.txt'
    }
    
    # Check available files
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
    
    # Analyze each device
    results = {}
    for device, filepath in available_files.items():
        result = load_and_analyze_device(filepath, device, max_lines=2000)
        if result:
            results[device] = result
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("DEVICE COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Device':<10} {'Records':<8} {'Trials':<8} {'Channels':<9} {'Valid Trials':<12}")
        print("-" * 55)
        
        for device, result in results.items():
            print(f"{device:<10} {result['n_records']:<8} {result['n_trials']:<8} "
                  f"{result['n_channels']:<9} {result['valid_trials']:<12}")
        
        # Create comparison visualization
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            devices = list(results.keys())
            
            # Plot 1: Number of records
            records = [results[d]['n_records'] for d in devices]
            axes[0, 0].bar(devices, records)
            axes[0, 0].set_title('Number of Records per Device')
            axes[0, 0].set_ylabel('Records')
            
            # Plot 2: Number of channels
            channels = [results[d]['n_channels'] for d in devices]
            axes[0, 1].bar(devices, channels)
            axes[0, 1].set_title('Number of Channels per Device')
            axes[0, 1].set_ylabel('Channels')
            
            # Plot 3: Signal mean comparison
            means = [results[d]['signal_stats']['mean'] for d in devices]
            axes[1, 0].bar(devices, means)
            axes[1, 0].set_title('Signal Mean per Device')
            axes[1, 0].set_ylabel('Mean Amplitude')
            
            # Plot 4: Valid trials
            valid_trials = [results[d]['valid_trials'] for d in devices]
            axes[1, 1].bar(devices, valid_trials)
            axes[1, 1].set_title('Valid Trials per Device')
            axes[1, 1].set_ylabel('Valid Trials')
            
            plt.tight_layout()
            plt.savefig('device_comparison_summary.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"\nComparison visualization saved: device_comparison_summary.png")
            
        except Exception as e:
            print(f"Comparison visualization error: {e}")
    
    # Save results
    try:
        import pickle
        with open('real_data_analysis_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to: real_data_analysis_results.pkl")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETED!")
    print(f"{'='*60}")
    
    print(f"\nSuccessfully analyzed {len(results)} devices:")
    for device, result in results.items():
        print(f"  - {device}: {result['n_records']} records, "
              f"{result['n_channels']} channels, "
              f"{result['valid_trials']} valid trials")
    
    print(f"\nGenerated files:")
    for device in results.keys():
        plot_file = f"{device.lower()}_analysis.png"
        if os.path.exists(plot_file):
            print(f"  - {plot_file}")
    if os.path.exists('device_comparison_summary.png'):
        print(f"  - device_comparison_summary.png")
    print(f"  - real_data_analysis_results.pkl")


if __name__ == "__main__":
    main()
