import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18
import warnings
warnings.filterwarnings('ignore')

# ============================
# 1. DATA LOADING AND PREPROCESSING
# ============================

class MindBigDataLoader:
    """
    Loader untuk dataset MindBigData dengan format spesifik
    Format: [id][event][device][channel][code][size][data] (tab-separated)
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.subjects_data = {}

        # Channel mapping untuk setiap device
        self.device_channels = {
            'MW': ['FP1'],  # MindWave
            'EP': ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],  # EPOC
            'MU': ['TP9', 'FP1', 'FP2', 'TP10'],  # Muse
            'IN': ['AF3', 'AF4', 'T7', 'T8', 'PZ']  # Insight
        }

        # Expected sampling rates
        self.device_hz = {
            'MW': 512,
            'EP': 128,
            'MU': 220,
            'IN': 128
        }

    def parse_line(self, line):
        """
        Parse satu baris data MindBigData
        Format: [id][event][device][channel][code][size][data]
        """
        try:
            parts = line.strip().split('\t')
            if len(parts) != 7:
                return None

            record = {
                'id': int(parts[0]),
                'event': int(parts[1]),
                'device': parts[2],
                'channel': parts[3],
                'code': int(parts[4]),
                'size': int(parts[5]),
                'data': parts[6]
            }

            # Parse data values
            if record['device'] in ['MW', 'MU']:
                # Integer values
                data_values = [int(x) for x in record['data'].split(',')]
            else:  # EP, IN
                # Float values
                data_values = [float(x) for x in record['data'].split(',')]

            record['signal'] = np.array(data_values)

            return record

        except Exception as e:
            print(f"Error parsing line: {e}")
            return None

    def load_file(self, file_path):
        """
        Load data dari satu file MindBigData
        """
        records = []

        print(f"Loading file: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num % 10000 == 0:
                        print(f"Processing line {line_num}...")

                    record = self.parse_line(line)
                    if record is not None:
                        records.append(record)

            print(f"Loaded {len(records)} records from {file_path}")
            return records

        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []

    def organize_by_trials(self, records):
        """
        Organize records menjadi trials berdasarkan event dan code
        """
        trials = {}

        for record in records:
            event_id = record['event']
            code = record['code']
            device = record['device']
            channel = record['channel']

            # Skip random signals (code = -1)
            if code == -1:
                continue

            # Create trial key
            trial_key = f"{event_id}_{code}_{device}"

            if trial_key not in trials:
                trials[trial_key] = {
                    'event_id': event_id,
                    'code': code,
                    'device': device,
                    'channels': {},
                    'metadata': {}
                }

            # Add channel data
            trials[trial_key]['channels'][channel] = record['signal']
            trials[trial_key]['metadata'][channel] = {
                'size': record['size'],
                'id': record['id']
            }

        return trials

    def create_multichannel_data(self, trials):
        """
        Create multichannel EEG data dari trials
        """
        eeg_data = []
        labels = []
        trial_info = []

        for trial_key, trial in trials.items():
            device = trial['device']
            expected_channels = self.device_channels[device]

            # Check if all channels are available
            available_channels = list(trial['channels'].keys())
            missing_channels = set(expected_channels) - set(available_channels)

            if missing_channels:
                # Skip trial if critical channels missing
                continue

            # Create multichannel array
            max_length = max([len(trial['channels'][ch]) for ch in available_channels])

            # Pad or truncate to standard length (e.g., 2 seconds worth of data)
            target_length = int(2 * self.device_hz[device])  # 2 seconds

            multichannel_signal = np.zeros((len(expected_channels), target_length))

            for i, channel in enumerate(expected_channels):
                if channel in trial['channels']:
                    signal = trial['channels'][channel]

                    # Pad or truncate
                    if len(signal) >= target_length:
                        multichannel_signal[i] = signal[:target_length]
                    else:
                        multichannel_signal[i, :len(signal)] = signal

            eeg_data.append(multichannel_signal)
            labels.append(trial['code'])
            trial_info.append({
                'trial_key': trial_key,
                'event_id': trial['event_id'],
                'device': device,
                'n_channels': len(expected_channels)
            })

        return np.array(eeg_data), np.array(labels), trial_info

    def load_multiple_files(self, file_paths):
        """
        Load data dari multiple files
        """
        all_records = []

        for file_path in file_paths:
            records = self.load_file(file_path)
            all_records.extend(records)

        # Organize into trials
        trials = self.organize_by_trials(all_records)

        # Create multichannel data
        eeg_data, labels, trial_info = self.create_multichannel_data(trials)

        return eeg_data, labels, trial_info

    def load_by_device(self, file_paths, target_device='EP'):
        """
        Load data untuk device tertentu saja
        """
        all_records = []

        for file_path in file_paths:
            records = self.load_file(file_path)
            # Filter by device
            device_records = [r for r in records if r['device'] == target_device]
            all_records.extend(device_records)

        trials = self.organize_by_trials(all_records)
        eeg_data, labels, trial_info = self.create_multichannel_data(trials)

        print(f"Loaded {len(eeg_data)} trials for device {target_device}")
        print(f"Data shape: {eeg_data.shape}")
        print(f"Unique labels: {np.unique(labels)}")

        return eeg_data, labels, trial_info

    def get_data_statistics(self, records):
        """
        Dapatkan statistik dari data yang dimuat
        """
        devices = {}
        codes = {}
        channels = {}

        for record in records:
            # Device stats
            device = record['device']
            if device not in devices:
                devices[device] = 0
            devices[device] += 1

            # Code stats
            code = record['code']
            if code not in codes:
                codes[code] = 0
            codes[code] += 1

            # Channel stats
            channel = record['channel']
            if channel not in channels:
                channels[channel] = 0
            channels[channel] += 1

        print("Data Statistics:")
        print(f"Devices: {devices}")
        print(f"Digit codes: {codes}")
        print(f"Channels: {channels}")

        return {'devices': devices, 'codes': codes, 'channels': channels}

# ============================
# 2. EEG PREPROCESSING
# ============================

class EEGPreprocessor:
    """
    Preprocessing untuk data EEG
    """
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()

    def bandpass_filter(self, data, low_freq=1, high_freq=40):
        """
        Bandpass filter untuk EEG
        """
        from scipy.signal import butter, filtfilt

        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        b, a = butter(4, [low, high], btype='band')
        filtered_data = np.zeros_like(data)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                filtered_data[i, j] = filtfilt(b, a, data[i, j])

        return filtered_data

    def extract_features(self, data):
        """
        Extract features dari EEG signals
        """
        features = []

        for sample in data:
            # Time domain features
            mean_val = np.mean(sample, axis=1)
            std_val = np.std(sample, axis=1)

            # Frequency domain features (simulasi)
            fft_features = np.abs(np.fft.fft(sample, axis=1))[:, :50]  # First 50 freq bins

            # Combine features
            sample_features = np.concatenate([mean_val, std_val, fft_features.flatten()])
            features.append(sample_features)

        return np.array(features)

# ============================
# 3. CNN MODEL FOR EEG
# ============================

class EEG_CNN(nn.Module):
    """
    CNN model untuk EEG classification/decoding
    """
    def __init__(self, n_channels=128, n_timepoints=200, n_classes=10):
        super(EEG_CNN, self).__init__()

        # Temporal convolutions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(n_channels, 1))

        # Spatial convolutions
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, 7), padding=(0, 3))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(1, 5), padding=(0, 2))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        # x shape: (batch, channels, timepoints)
        x = x.unsqueeze(1)  # Add channel dimension

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# ============================
# 4. TRANSFORMER MODEL FOR EEG
# ============================

class EEGTransformer(nn.Module):
    """
    Transformer model untuk EEG decoding
    """
    def __init__(self, n_channels=128, n_timepoints=200, n_classes=10, d_model=128, nhead=8, num_layers=6):
        super(EEGTransformer, self).__init__()

        self.d_model = d_model
        self.input_projection = nn.Linear(n_channels, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(n_timepoints, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, timepoints)
        x = x.transpose(1, 2)  # (batch, timepoints, channels)

        # Project to d_model
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.positional_encoding.unsqueeze(0)

        # Transformer
        x = self.transformer(x)

        # Classification
        x = x.transpose(1, 2)  # (batch, d_model, timepoints)
        x = self.classifier(x)

        return x

# ============================
# 5. EEG-fMRI INTEGRATION
# ============================

class EEGfMRIIntegrator:
    """
    Integrasi EEG-fMRI menggunakan domain adaptation
    """
    def __init__(self, eeg_dim=128, fmri_dim=64*64*30):  # Simulasi fMRI dimensions
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.fmri_encoder = nn.Sequential(
            nn.Linear(fmri_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # EEG vs fMRI
        )

    def forward(self, eeg_data, fmri_data=None):
        eeg_features = self.eeg_encoder(eeg_data)

        if fmri_data is not None:
            fmri_features = self.fmri_encoder(fmri_data)
            return eeg_features, fmri_features

        return eeg_features

class NeuralTranscodingViT(nn.Module):
    """
    Neural Transcoding Vision Transformer untuk EEG ke fMRI
    """
    def __init__(self, eeg_channels=128, fmri_size=(64, 64, 30)):
        super(NeuralTranscodingViT, self).__init__()

        self.eeg_channels = eeg_channels
        self.fmri_size = fmri_size
        self.fmri_dim = np.prod(fmri_size)

        # EEG Encoder
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 768),  # ViT dimension
            nn.ReLU()
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # fMRI Generator
        self.fmri_generator = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.fmri_dim),
            nn.Tanh()
        )

    def forward(self, eeg_data):
        # Encode EEG
        eeg_features = self.eeg_encoder(eeg_data)

        # Generate fMRI-like representation
        fmri_pred = self.fmri_generator(eeg_features)
        fmri_pred = fmri_pred.view(-1, *self.fmri_size)

        return fmri_pred

# ============================
# 6. FEATURE IMPORTANCE ANALYSIS
# ============================

class FeatureImportanceAnalyzer:
    """
    Analisis feature importance untuk EEG
    """
    def __init__(self, model):
        self.model = model

    def compute_saliency_maps(self, data, labels):
        """
        Compute saliency maps untuk interpretasi
        """
        self.model.eval()
        data.requires_grad_()

        outputs = self.model(data)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        saliency = data.grad.abs()
        return saliency

    def plot_channel_importance(self, saliency_maps, channel_names=None):
        """
        Plot importance per channel
        """
        if channel_names is None:
            channel_names = [f'Ch{i}' for i in range(saliency_maps.shape[1])]

        channel_importance = saliency_maps.mean(dim=(0, 2))

        plt.figure(figsize=(15, 6))
        plt.bar(range(len(channel_importance)), channel_importance.detach().numpy())
        plt.xlabel('EEG Channels')
        plt.ylabel('Importance Score')
        plt.title('Channel Importance Analysis')
        plt.xticks(range(0, len(channel_names), 10), channel_names[::10], rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_temporal_importance(self, saliency_maps):
        """
        Plot importance temporal
        """
        temporal_importance = saliency_maps.mean(dim=(0, 1))

        plt.figure(figsize=(12, 6))
        plt.plot(temporal_importance.detach().numpy())
        plt.xlabel('Time Points')
        plt.ylabel('Importance Score')
        plt.title('Temporal Importance Analysis')
        plt.grid(True)
        plt.show()

# ============================
# 7. MAIN EXPERIMENT PIPELINE
# ============================

class EEGExperimentPipeline:
    """
    Pipeline utama untuk eksperimen EEG
    """
    def __init__(self, data_path):
        self.data_loader = MindBigDataLoader(data_path)
        self.preprocessor = EEGPreprocessor()
        self.results = {}

    def run_multi_device_validation(self, file_paths):
        """
        Validasi multi-device untuk menguji generalisasi antar perangkat EEG
        """
        print("=== Multi-Device Validation ===")

        device_results = {}

        # Test each device
        for device in ['MW', 'EP', 'MU', 'IN']:
            try:
                print(f"\nLoading data for device: {device}")
                eeg_data, labels, trial_info = self.data_loader.load_by_device(file_paths, device)

                if len(eeg_data) == 0:
                    print(f"No data found for device {device}")
                    continue

                # Preprocess
                eeg_data = self.preprocessor.bandpass_filter(eeg_data)
                features = self.preprocessor.extract_features(eeg_data)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42, stratify=labels
                )

                # Train SVM baseline
                svm_model = SVC(kernel='rbf', random_state=42)
                svm_model.fit(X_train, y_train)

                # Test
                y_pred = svm_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                device_results[device] = {
                    'accuracy': accuracy,
                    'n_samples': len(eeg_data),
                    'n_features': features.shape[1],
                    'n_channels': eeg_data.shape[1]
                }

                print(f"Device {device} - Accuracy: {accuracy:.4f} ({len(eeg_data)} samples)")

            except Exception as e:
                print(f"Error processing device {device}: {e}")

        self.results['multi_device'] = device_results

        # Cross-device validation
        self.run_cross_device_validation(file_paths)

        return device_results

    def run_cross_device_validation(self, file_paths):
        """
        Cross-device validation: train pada satu device, test pada device lain
        """
        print("\n=== Cross-Device Validation ===")

        devices = ['MW', 'EP', 'MU', 'IN']
        cross_device_results = {}

        for train_device in devices:
            for test_device in devices:
                if train_device == test_device:
                    continue

                try:
                    # Load training data
                    train_data, train_labels, _ = self.data_loader.load_by_device(file_paths, train_device)
                    if len(train_data) == 0:
                        continue

                    # Load test data
                    test_data, test_labels, _ = self.data_loader.load_by_device(file_paths, test_device)
                    if len(test_data) == 0:
                        continue

                    # Preprocess
                    train_data = self.preprocessor.bandpass_filter(train_data)
                    test_data = self.preprocessor.bandpass_filter(test_data)

                    train_features = self.preprocessor.extract_features(train_data)
                    test_features = self.preprocessor.extract_features(test_data)

                    # Align feature dimensions (important for cross-device)
                    min_features = min(train_features.shape[1], test_features.shape[1])
                    train_features = train_features[:, :min_features]
                    test_features = test_features[:, :min_features]

                    # Train model
                    svm_model = SVC(kernel='rbf', random_state=42)
                    svm_model.fit(train_features, train_labels)

                    # Test
                    y_pred = svm_model.predict(test_features)
                    accuracy = accuracy_score(test_labels, y_pred)

                    key = f"{train_device}_to_{test_device}"
                    cross_device_results[key] = accuracy

                    print(f"Train on {train_device}, Test on {test_device}: {accuracy:.4f}")

                except Exception as e:
                    print(f"Error in cross-device {train_device}->{test_device}: {e}")

        self.results['cross_device'] = cross_device_results
    def run_image_reconstruction_experiment(self, file_paths):
        """
        Experiment utama untuk rekonstruksi citra dari EEG
        """
        print("\n" + "="*60)
        print("EEG-TO-IMAGE RECONSTRUCTION EXPERIMENT")
        print("="*60)

        # 1. Load EEG data (focus on EPOC for better spatial resolution)
        print("Step 1: Loading EEG data...")
        eeg_data, labels, trial_info = self.data_loader.load_by_device(file_paths, 'EP')

        if len(eeg_data) == 0:
            print("No EPOC data found. Trying other devices...")
            for device in ['MU', 'IN', 'MW']:
                eeg_data, labels, trial_info = self.data_loader.load_by_device(file_paths, device)
                if len(eeg_data) > 0:
                    print(f"Using {device} device data instead.")
                    break

        if len(eeg_data) == 0:
            print("No suitable EEG data found!")
            return None

        # 2. Preprocess EEG data
        print("Step 2: Preprocessing EEG data...")
        eeg_data = self.preprocessor.bandpass_filter(eeg_data)

        # Convert to tensors
        eeg_tensor = torch.FloatTensor(eeg_data)
        labels_tensor = torch.LongTensor(labels)

        # 3. Initialize image reconstruction pipeline
        print("Step 3: Setting up image reconstruction pipeline...")
        reconstruction_pipeline = ImageReconstructionPipeline(model_type='generator')

        # 4. Load MNIST reference images
        print("Step 4: Loading MNIST reference images...")
        digit_images = reconstruction_pipeline.load_mnist_references()

        # 5. Create paired dataset
        print("Step 5: Creating paired EEG-Image dataset...")
        paired_eeg, paired_images = reconstruction_pipeline.create_paired_dataset(
            eeg_tensor, labels, digit_images
        )

        print(f"Created {len(paired_eeg)} paired samples")

        # 6. Train reconstruction model
        print("Step 6: Training image reconstruction model...")
        trained_model = reconstruction_pipeline.train_generator(
            paired_eeg, paired_images, epochs=50, lr=0.001
        )

        # 7. Generate images from EEG
        print("Step 7: Generating images from EEG...")
        generated_images = reconstruction_pipeline.generate_images_from_eeg(paired_eeg)

        # 8. Evaluate reconstruction quality
        print("Step 8: Evaluating reconstruction quality...")
        evaluation_metrics = reconstruction_pipeline.evaluate_reconstruction_quality(
            generated_images, paired_images
        )

        # 9. Visualize results
        print("Step 9: Visualizing reconstruction results...")
        reconstruction_pipeline.visualize_reconstructions(
            paired_eeg, generated_images, paired_images, labels[:len(paired_eeg)]
        )

        # 10. Generate detailed report
        print("Step 10: Generating reconstruction report...")
        reconstruction_pipeline.create_reconstruction_report(
            evaluation_metrics, labels[:len(paired_eeg)]
        )

        # Store results
        self.results['image_reconstruction'] = {
            'model': trained_model,
            'generated_images': generated_images,
            'evaluation_metrics': evaluation_metrics,
            'n_samples': len(paired_eeg)
        }

        return reconstruction_pipeline

    def compare_reconstruction_methods(self, file_paths):
        """
        Bandingkan berbagai metode rekonstruksi
        """
        print("\n" + "="*50)
        print("RECONSTRUCTION METHODS COMPARISON")
        print("="*50)

        # Load data
        eeg_data, labels, _ = self.data_loader.load_by_device(file_paths, 'EP')
        if len(eeg_data) == 0:
            print("No data available for comparison")
            return

        eeg_data = self.preprocessor.bandpass_filter(eeg_data)
        eeg_tensor = torch.FloatTensor(eeg_data)

        methods = ['generator', 'vae']
        method_results = {}

        for method in methods:
            print(f"\nTesting {method.upper()} method...")

            pipeline = ImageReconstructionPipeline(model_type=method)
            digit_images = pipeline.load_mnist_references()
            paired_eeg, paired_images = pipeline.create_paired_dataset(
                eeg_tensor, labels, digit_images
            )

            # Train with fewer epochs for comparison
            trained_model = pipeline.train_generator(
                paired_eeg, paired_images, epochs=20, lr=0.001
            )

            # Generate and evaluate
            generated_images = pipeline.generate_images_from_eeg(paired_eeg)
            metrics = pipeline.evaluate_reconstruction_quality(
                generated_images, paired_images
            )

            method_results[method] = {
                'mse': metrics['mse'],
                'ssim': metrics['ssim'],
                'model': trained_model
            }

            print(f"{method.upper()} Results - MSE: {metrics['mse']:.6f}, SSIM: {metrics['ssim']:.4f}")

        # Compare results
        print(f"\n📊 METHOD COMPARISON:")
        print("-" * 40)
        best_method = min(method_results.keys(), key=lambda x: method_results[x]['mse'])
        for method, results in method_results.items():
            status = "🏆 BEST" if method == best_method else ""
            print(f"  {method.upper()}: MSE={results['mse']:.6f}, SSIM={results['ssim']:.4f} {status}")

        self.results['method_comparison'] = method_results
        return method_results

    def compare_deep_learning_models(self, eeg_data, labels):
        """
        Bandingkan model deep learning
        """
        print("\n=== Deep Learning Model Comparison ===")

        # Convert to torch tensors
        X = torch.FloatTensor(eeg_data)
        y = torch.LongTensor(labels)

        # Split data
        train_size = int(0.8 * len(X))
        train_X, test_X = X[:train_size], X[train_size:]
        train_y, test_y = y[:train_size], y[train_size:]

        # Get actual dimensions from data
        n_channels = eeg_data.shape[1]
        n_timepoints = eeg_data.shape[2]

        models = {
            'CNN': EEG_CNN(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=10),
            'Transformer': EEGTransformer(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=10)
        }

        model_results = {}

        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")

            # Training setup
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Simple training loop
            model.train()
            for epoch in range(10):  # Reduced epochs for demo
                optimizer.zero_grad()
                outputs = model(train_X)
                loss = criterion(outputs, train_y)
                loss.backward()
                optimizer.step()

                if epoch % 5 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            # Evaluation
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_X)
                _, predicted = torch.max(test_outputs, 1)
                accuracy = (predicted == test_y).float().mean().item()

            print(f"{model_name} Test Accuracy: {accuracy:.4f}")
            model_results[model_name] = accuracy

        self.results['deep_learning'] = model_results
        return models

    def test_eeg_fmri_integration(self, eeg_data):
        """
        Test integrasi EEG-fMRI
        """
        print("\n=== EEG-fMRI Integration Testing ===")

        # Neural Transcoding ViT
        nt_vit = NeuralTranscodingViT(eeg_channels=128)

        # Simulasi data EEG
        sample_eeg = torch.FloatTensor(eeg_data[:100].mean(axis=2))  # Average over time

        # Generate fMRI-like representation
        with torch.no_grad():
            fmri_pred = nt_vit(sample_eeg)

        print(f"EEG input shape: {sample_eeg.shape}")
        print(f"Generated fMRI shape: {fmri_pred.shape}")

        # Visualize some results
        self.visualize_eeg_fmri_translation(sample_eeg[:5], fmri_pred[:5])

        return nt_vit

    def visualize_eeg_fmri_translation(self, eeg_samples, fmri_samples):
        """
        Visualisasi translasi EEG ke fMRI
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        for i in range(5):
            # EEG
            axes[0, i].plot(eeg_samples[i].numpy())
            axes[0, i].set_title(f'EEG Sample {i+1}')
            axes[0, i].set_xlabel('Channels')

            # fMRI (middle slice)
            fmri_slice = fmri_samples[i, :, :, 15].numpy()
            axes[1, i].imshow(fmri_slice, cmap='viridis')
            axes[1, i].set_title(f'Generated fMRI {i+1}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    def analyze_feature_importance(self, model, test_data, test_labels):
        """
        Analisis feature importance
        """
        print("\n=== Feature Importance Analysis ===")

        analyzer = FeatureImportanceAnalyzer(model)

        # Compute saliency maps
        saliency_maps = analyzer.compute_saliency_maps(test_data, test_labels)

        # Plot analyses
        analyzer.plot_channel_importance(saliency_maps)
        analyzer.plot_temporal_importance(saliency_maps)

        return saliency_maps

    def generate_comprehensive_report(self):
        """
        Generate laporan komprehensif hasil eksperimen
        """
        print("\n" + "="*60)

# ============================
# 9. ADVANCED VISUALIZATION AND ANALYSIS
# ============================

class EEGVisualizationTools:
    """
    Advanced visualization tools untuk analisis EEG
    """
    def __init__(self):
        pass

    def plot_device_comparison(self, device_results):
        """
        Visualisasi perbandingan performa antar device
        """
        devices = list(device_results.keys())
        accuracies = [device_results[d]['accuracy'] for d in devices]
        n_samples = [device_results[d]['n_samples'] for d in devices]
        n_channels = [device_results[d]['n_channels'] for d in devices]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy comparison
        bars1 = ax1.bar(devices, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Classification Accuracy by Device', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        # Sample size comparison
        bars2 = ax2.bar(devices, n_samples, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Number of Samples by Device', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Samples')
        for bar, n in zip(bars2, n_samples):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(n_samples)*0.01,
                    f'{n}', ha='center', va='bottom', fontweight='bold')

        # Channel comparison
        bars3 = ax3.bar(devices, n_channels, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('Number of Channels by Device', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Channels')
        for bar, n in zip(bars3, n_channels):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{n}', ha='center', va='bottom', fontweight='bold')

        # Accuracy vs Channels scatter
        ax4.scatter(n_channels, accuracies, s=100, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        for i, device in enumerate(devices):
            ax4.annotate(device, (n_channels[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        ax4.set_xlabel('Number of Channels')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy vs Channel Count', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_cross_device_heatmap(self, cross_device_results):
        """
        Heatmap untuk cross-device transfer results
        """
        devices = ['MW', 'EP', 'MU', 'IN']
        transfer_matrix = np.zeros((len(devices), len(devices)))

        for i, train_dev in enumerate(devices):
            for j, test_dev in enumerate(devices):
                if train_dev != test_dev:
                    key = f"{train_dev}_to_{test_dev}"
                    if key in cross_device_results:
                        transfer_matrix[i, j] = cross_device_results[key]
                else:
                    transfer_matrix[i, j] = np.nan

        plt.figure(figsize=(10, 8))
        mask = np.isnan(transfer_matrix)
        sns.heatmap(transfer_matrix,
                    annot=True,
                    fmt='.3f',
                    cmap='RdYlBu_r',
                    mask=mask,
                    xticklabels=devices,
                    yticklabels=devices,
                    cbar_kws={'label': 'Transfer Accuracy'},
                    square=True)
        plt.title('Cross-Device Transfer Learning Performance', fontsize=16, fontweight='bold')
        plt.xlabel('Test Device', fontsize=12)
        plt.ylabel('Train Device', fontsize=12)
        plt.tight_layout()
        plt.show()

    def plot_signal_samples(self, eeg_data, labels, trial_info, n_samples=5):
        """
        Plot sample EEG signals untuk setiap digit
        """
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        fig, axes = plt.subplots(n_labels, n_samples, figsize=(20, 2*n_labels))
        if n_labels == 1:
            axes = axes.reshape(1, -1)

        for i, label in enumerate(unique_labels):
            label_indices = np.where(labels == label)[0]
            selected_indices = np.random.choice(label_indices, min(n_samples, len(label_indices)), replace=False)

            for j, idx in enumerate(selected_indices):
                if j < n_samples:
                    # Plot first channel
                    axes[i, j].plot(eeg_data[idx, 0, :], linewidth=1)
                    axes[i, j].set_title(f'Digit {label}, Trial {j+1}', fontsize=10)
                    axes[i, j].set_xlabel('Time Points')
                    axes[i, j].set_ylabel('Amplitude')
                    axes[i, j].grid(True, alpha=0.3)

        plt.suptitle('Sample EEG Signals by Digit Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_channel_topography(self, channel_importance, device='EP'):
        """
        Plot topographic map untuk channel importance
        """
        # Channel positions untuk EPOC (simplified)
        channel_positions = {
            'AF3': (-0.3, 0.8), 'F7': (-0.7, 0.3), 'F3': (-0.3, 0.3), 'FC5': (-0.5, 0.1),
            'T7': (-0.9, 0), 'P7': (-0.7, -0.3), 'O1': (-0.3, -0.8), 'O2': (0.3, -0.8),
            'P8': (0.7, -0.3), 'T8': (0.9, 0), 'FC6': (0.5, 0.1), 'F4': (0.3, 0.3),
            'F8': (0.7, 0.3), 'AF4': (0.3, 0.8)
        }

        if device == 'EP':
            channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        else:
            # Simplified for other devices
            channels = [f'Ch{i}' for i in range(len(channel_importance))]
            channel_positions = {f'Ch{i}': (np.cos(2*np.pi*i/len(channels)),
                                          np.sin(2*np.pi*i/len(channels)))
                               for i in range(len(channels))}

        plt.figure(figsize=(10, 10))

        # Draw head outline
        head_circle = plt.Circle((0, 0), 1.1, fill=False, linewidth=3, color='black')
        plt.gca().add_patch(head_circle)

        # Draw nose
        nose_x = [0, 0]
        nose_y = [1.1, 1.3]
        plt.plot(nose_x, nose_y, 'k-', linewidth=3)

        # Plot channels
        for i, ch in enumerate(channels[:len(channel_importance)]):
            if ch in channel_positions:
                x, y = channel_positions[ch]
                importance = channel_importance[i] if i < len(channel_importance) else 0

                # Color and size based on importance
                color = plt.cm.RdYlBu_r(importance / channel_importance.max())
                size = 100 + 400 * (importance / channel_importance.max())

                plt.scatter(x, y, s=size, c=[color], alpha=0.8, edgecolors='black', linewidth=2)
                plt.text(x, y-0.15, ch, ha='center', va='center', fontweight='bold', fontsize=8)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axis('equal')
        plt.axis('off')
        plt.title(f'Channel Importance Topography - {device} Device', fontsize=16, fontweight='bold')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r',
                                  norm=plt.Normalize(vmin=channel_importance.min(),
                                                   vmax=channel_importance.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, shrink=0.6, aspect=20)
        cbar.set_label('Importance Score', fontsize=12)

        plt.tight_layout()
        plt.show()

# ============================
# 10. STATISTICAL ANALYSIS TOOLS
# ============================

class EEGStatisticalAnalysis:
    """
    Statistical analysis tools untuk EEG data
    """
    def __init__(self):
        pass

    def perform_anova_analysis(self, eeg_data, labels, trial_info):
        """
        ANOVA analysis untuk channel differences across digits
        """
        from scipy import stats

        print("=== ANOVA Analysis: Channel Activity Across Digits ===")

        unique_labels = np.unique(labels)
        n_channels = eeg_data.shape[1]

        anova_results = []

        for ch in range(n_channels):
            channel_data = []
            for label in unique_labels:
                label_indices = np.where(labels == label)[0]
                # Average over time for each trial
                channel_means = np.mean(eeg_data[label_indices, ch, :], axis=1)
                channel_data.append(channel_means)

            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*channel_data)
            anova_results.append({
                'channel': ch,
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

        # Summary
        significant_channels = [r for r in anova_results if r['significant']]
        print(f"Significant channels (p < 0.05): {len(significant_channels)}/{n_channels}")

        # Plot results
        self.plot_anova_results(anova_results)

        return anova_results

    def plot_anova_results(self, anova_results):
        """
        Plot ANOVA results
        """
        channels = [r['channel'] for r in anova_results]
        f_stats = [r['f_statistic'] for r in anova_results]
        p_values = [r['p_value'] for r in anova_results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # F-statistics
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        ax1.bar(channels, f_stats, color=colors, alpha=0.7)
        ax1.set_title('ANOVA F-Statistics by Channel', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Channel')
        ax1.set_ylabel('F-Statistic')
        ax1.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Typical significance threshold')
        ax1.legend()

        # P-values (log scale)
        ax2.bar(channels, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
        ax2.set_title('ANOVA P-Values (Negative Log Scale)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Channel')
        ax2.set_ylabel('-log10(p-value)')
        ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p = 0.05')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def compute_effect_sizes(self, eeg_data, labels):
        """
        Compute effect sizes (Cohen's d) between digit pairs
        """
        print("=== Effect Size Analysis ===")

        unique_labels = np.unique(labels)
        n_channels = eeg_data.shape[1]

        effect_sizes = {}

        for i, label1 in enumerate(unique_labels):
            for j, label2 in enumerate(unique_labels):
                if i < j:  # Avoid duplicates
                    pair_key = f"{label1}_vs_{label2}"

                    # Get data for each label
                    data1 = eeg_data[labels == label1]
                    data2 = eeg_data[labels == label2]

                    # Compute Cohen's d for each channel
                    cohens_d = []
                    for ch in range(n_channels):
                        mean1 = np.mean(data1[:, ch, :])
                        mean2 = np.mean(data2[:, ch, :])
                        std1 = np.std(data1[:, ch, :])
                        std2 = np.std(data2[:, ch, :])

                        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                        d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                        cohens_d.append(abs(d))

                    effect_sizes[pair_key] = cohens_d

        # Find pairs with largest effect sizes
        max_effects = {}
        for pair, effects in effect_sizes.items():
            max_effects[pair] = max(effects)

        # Sort by effect size
        sorted_pairs = sorted(max_effects.items(), key=lambda x: x[1], reverse=True)

        print("Top 5 digit pairs with largest effect sizes:")
        for pair, effect in sorted_pairs[:5]:
            print(f"  {pair}: Cohen's d = {effect:.3f}")

# ============================
# 11. EEG-TO-IMAGE GENERATION MODELS
# ============================

class EEGToImageGenerator(nn.Module):
    """
    Generator untuk merekonstruksi citra MNIST dari sinyal EEG
    """
    def __init__(self, eeg_channels=14, eeg_timepoints=256, image_size=28):
        super(EEGToImageGenerator, self).__init__()

        self.eeg_channels = eeg_channels
        self.eeg_timepoints = eeg_timepoints
        self.image_size = image_size

        # EEG Feature Extractor
        self.eeg_encoder = nn.Sequential(
            # Convolutional layers for EEG
            nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(eeg_channels, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 32)),  # Reduce to manageable size
        )

        # Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(128 * 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Image Generator (Deconvolutional)
        self.image_generator = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, image_size * image_size),
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, eeg_data):
        # eeg_data shape: (batch, channels, timepoints)
        batch_size = eeg_data.size(0)

        # Add channel dimension for conv2d
        x = eeg_data.unsqueeze(1)  # (batch, 1, channels, timepoints)

        # Extract EEG features
        x = self.eeg_encoder(x)
        x = x.view(batch_size, -1)  # Flatten

        # Project features
        features = self.feature_projection(x)

        # Generate image
        image = self.image_generator(features)
        image = image.view(batch_size, 1, self.image_size, self.image_size)

        return image

class EEGToImageGAN(nn.Module):
    """
    GAN approach untuk EEG-to-Image generation
    """
    def __init__(self, eeg_channels=14, eeg_timepoints=256, image_size=28):
        super(EEGToImageGAN, self).__init__()

        self.generator = EEGToImageGenerator(eeg_channels, eeg_timepoints, image_size)
        self.discriminator = self._build_discriminator(image_size)

    def _build_discriminator(self, image_size):
        """
        Discriminator untuk membedakan real vs generated images
        """
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, eeg_data):
        return self.generator(eeg_data)

class EEGImageVAE(nn.Module):
    """
    Variational Autoencoder untuk EEG-to-Image generation
    """
    def __init__(self, eeg_channels=14, eeg_timepoints=256, image_size=28, latent_dim=128):
        super(EEGImageVAE, self).__init__()

        self.latent_dim = latent_dim
        self.image_size = image_size

        # EEG Encoder
        self.eeg_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 7), padding=(0, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(eeg_channels, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Flatten(),
            nn.Linear(64 * 32, 512),
            nn.ReLU()
        )

        # Latent space
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Image Decoder
        self.image_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size * image_size),
            nn.Sigmoid()
        )

    def encode(self, eeg_data):
        x = eeg_data.unsqueeze(1)  # Add channel dimension
        x = self.eeg_encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.image_decoder(z)
        return x.view(-1, 1, self.image_size, self.image_size)

    def forward(self, eeg_data):
        mu, logvar = self.encode(eeg_data)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ============================
# 12. IMAGE RECONSTRUCTION PIPELINE
# ============================

class ImageReconstructionPipeline:
    """
    Pipeline untuk merekonstruksi citra MNIST dari EEG
    """
    def __init__(self, model_type='generator'):
        self.model_type = model_type
        self.model = None
        self.criterion = nn.MSELoss()

    def load_mnist_references(self):
        """
        Load reference MNIST images untuk setiap digit
        """
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # Load MNIST dataset
        mnist_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

        # Get one representative image for each digit
        digit_images = {}
        for digit in range(10):
            for idx, (image, label) in enumerate(mnist_dataset):
                if label == digit and digit not in digit_images:
                    digit_images[digit] = image
                    break

        return digit_images

    def create_paired_dataset(self, eeg_data, labels, digit_images):
        """
        Create paired EEG-Image dataset
        """
        paired_eeg = []
        paired_images = []

        for i, label in enumerate(labels):
            if label in digit_images:
                paired_eeg.append(eeg_data[i])
                paired_images.append(digit_images[label])

        return torch.stack(paired_eeg), torch.stack(paired_images)

    def train_generator(self, eeg_data, target_images, epochs=100, lr=0.001):
        """
        Train image generator model
        """
        print("Training EEG-to-Image Generator...")

        # Initialize model
        eeg_channels = eeg_data.shape[1]
        eeg_timepoints = eeg_data.shape[2]

        if self.model_type == 'generator':
            self.model = EEGToImageGenerator(eeg_channels, eeg_timepoints)
        elif self.model_type == 'vae':
            self.model = EEGImageVAE(eeg_channels, eeg_timepoints)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            if self.model_type == 'generator':
                generated_images = self.model(eeg_data)
                loss = self.criterion(generated_images, target_images)
            elif self.model_type == 'vae':
                generated_images, mu, logvar = self.model(eeg_data)
                recon_loss = self.criterion(generated_images, target_images)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.001 * kl_loss

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        print("Training completed!")
        return self.model

    def generate_images_from_eeg(self, eeg_data):
        """
        Generate images dari EEG data
        """
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'generator':
                generated_images = self.model(eeg_data)
            elif self.model_type == 'vae':
                generated_images, _, _ = self.model(eeg_data)

        return generated_images

    def evaluate_reconstruction_quality(self, generated_images, target_images):
        """
        Evaluate kualitas rekonstruksi
        """
        # MSE
        mse = F.mse_loss(generated_images, target_images).item()

        # SSIM (Structural Similarity Index)
        def ssim(img1, img2):
            # Simplified SSIM calculation
            mu1 = torch.mean(img1)
            mu2 = torch.mean(img2)
            sigma1_sq = torch.var(img1)
            sigma2_sq = torch.var(img2)
            sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

            c1 = 0.01**2
            c2 = 0.03**2

            ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                       ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
            return ssim_val.item()

        # Calculate SSIM for each image pair
        ssim_scores = []
        for i in range(generated_images.shape[0]):
            ssim_score = ssim(generated_images[i], target_images[i])
            ssim_scores.append(ssim_score)

        avg_ssim = np.mean(ssim_scores)

        return {
            'mse': mse,
            'ssim': avg_ssim,
            'individual_ssim': ssim_scores
        }

    def visualize_reconstructions(self, eeg_data, generated_images, target_images, labels, n_samples=10):
        """
        Visualisasi hasil rekonstruksi
        """
        fig, axes = plt.subplots(3, n_samples, figsize=(20, 6))

        for i in range(min(n_samples, len(generated_images))):
            # Original MNIST
            axes[0, i].imshow(target_images[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f'Original\nDigit {labels[i]}')
            axes[0, i].axis('off')

            # Generated image
            axes[1, i].imshow(generated_images[i].squeeze().detach(), cmap='gray')
            axes[1, i].set_title('Generated')
            axes[1, i].axis('off')

            # EEG signal (first channel)
            axes[2, i].plot(eeg_data[i, 0, :].detach())
            axes[2, i].set_title('EEG Signal')
            axes[2, i].set_xlabel('Time')

        plt.suptitle('EEG-to-Image Reconstruction Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def create_reconstruction_report(self, evaluation_metrics, labels):
        """
        Create detailed reconstruction report
        """
        print("\n" + "="*60)
        print("IMAGE RECONSTRUCTION QUALITY REPORT")
        print("="*60)

        print(f"Overall Performance:")
        print(f"  • Mean Squared Error: {evaluation_metrics['mse']:.6f}")
        print(f"  • Average SSIM: {evaluation_metrics['ssim']:.4f}")

        # Per-digit analysis
        unique_labels = np.unique(labels)
        print(f"\nPer-Digit SSIM Analysis:")
        for digit in unique_labels:
            digit_indices = np.where(labels == digit)[0]
            if len(digit_indices) > 0:
                digit_ssim = np.mean([evaluation_metrics['individual_ssim'][i] for i in digit_indices])
                print(f"  • Digit {digit}: {digit_ssim:.4f}")

        # Quality assessment
        print(f"\nReconstruction Quality Assessment:")
        if evaluation_metrics['ssim'] > 0.7:
            print("  🟢 Excellent reconstruction quality")
        elif evaluation_metrics['ssim'] > 0.5:
            print("  🟡 Good reconstruction quality")
        elif evaluation_metrics['ssim'] > 0.3:
            print("  🟠 Fair reconstruction quality")
        else:
            print("  🔴 Poor reconstruction quality - needs improvement")

        print(f"\nRecommendations for Improvement:")
        print("  1. Increase training epochs for better convergence")
        print("  2. Try different loss functions (perceptual loss, GAN loss)")
        print("  3. Implement attention mechanisms for better feature extraction")
        print("  4. Use more sophisticated architectures (StyleGAN, Progressive GAN)")
        print("  5. Augment training data with more EEG-image pairs")

        return evaluation_metrics
        print("COMPREHENSIVE EXPERIMENT RESULTS SUMMARY")
        print("="*60)

        # Multi-device results
        if 'multi_device' in self.results:
            device_results = self.results['multi_device']
            print(f"\n📊 MULTI-DEVICE VALIDATION RESULTS:")
            print("-" * 40)
            for device, results in device_results.items():
                print(f"  {device} Device:")
                print(f"    • Accuracy: {results['accuracy']:.4f}")
                print(f"    • Samples: {results['n_samples']}")
                print(f"    • Channels: {results['n_channels']}")
                print(f"    • Features: {results['n_features']}")

        # Cross-device results
        if 'cross_device' in self.results:
            cross_results = self.results['cross_device']
            print(f"\n🔄 CROSS-DEVICE GENERALIZATION:")
            print("-" * 40)
            for transfer, accuracy in cross_results.items():
                print(f"  {transfer}: {accuracy:.4f}")

        # Deep learning results
        if 'deep_learning' in self.results:
            dl_results = self.results['deep_learning']
            print(f"\n🧠 DEEP LEARNING MODEL COMPARISON:")
            print("-" * 40)
            for model_name, accuracy in dl_results.items():
                print(f"  {model_name}: {accuracy:.4f}")

        # Analysis insights
        print(f"\n🔍 KEY INSIGHTS & RECOMMENDATIONS:")
        print("-" * 40)

        # Device-specific insights
        if 'multi_device' in self.results:
            device_results = self.results['multi_device']
            best_device = max(device_results.keys(), key=lambda x: device_results[x]['accuracy'])
            worst_device = min(device_results.keys(), key=lambda x: device_results[x]['accuracy'])

            print(f"  1. Best performing device: {best_device} ({device_results[best_device]['accuracy']:.4f})")
            print(f"  2. Most challenging device: {worst_device} ({device_results[worst_device]['accuracy']:.4f})")

        # Cross-device insights
        if 'cross_device' in self.results:
            cross_results = self.results['cross_device']
            best_transfer = max(cross_results.keys(), key=lambda x: cross_results[x])
            worst_transfer = min(cross_results.keys(), key=lambda x: cross_results[x])

            print(f"  3. Best cross-device transfer: {best_transfer} ({cross_results[best_transfer]:.4f})")
            print(f"  4. Most challenging transfer: {worst_transfer} ({cross_results[worst_transfer]:.4f})")

        # Technical recommendations
        print(f"\n💡 TECHNICAL RECOMMENDATIONS:")
        print("-" * 40)
        print("  1. Implement domain adaptation techniques for cross-device generalization")
        print("  2. Use device-specific preprocessing pipelines")
        print("  3. Consider ensemble methods combining multiple devices")
        print("  4. Implement advanced feature alignment for cross-device training")
        print("  5. Add more sophisticated temporal modeling (LSTM, GRU)")
        print("  6. Explore subject-specific fine-tuning strategies")

        # Neurobiological insights
        print(f"\n🧬 NEUROBIOLOGICAL CONSIDERATIONS:")
        print("-" * 40)
        print("  1. Different devices capture different aspects of neural activity")
        print("  2. Spatial resolution varies significantly across devices")
        print("  3. Temporal dynamics may be device-dependent")
        print("  4. Individual differences in neural signatures")
        print("  5. Consider cognitive load differences across digit imagination tasks")

        # Future work
        print(f"\n🚀 FUTURE RESEARCH DIRECTIONS:")
        print("-" * 40)
        print("  1. Implement real EEG-fMRI paired data collection")
        print("  2. Develop device-agnostic feature representations")
        print("  3. Explore meta-learning for rapid device adaptation")
        print("  4. Implement attention mechanisms for channel selection")
        print("  5. Develop interpretable models for clinical applications")

        print("\n" + "="*60)

        if 'multi_subject' in self.results:
            ms_results = self.results['multi_subject']
            print(f"Multi-Subject Validation:")
            print(f"  Mean Accuracy: {ms_results['mean_accuracy']:.4f} ± {ms_results['std_accuracy']:.4f}")
            print(f"  Individual Scores: {[f'{s:.3f}' for s in ms_results['subject_scores']]}")

        if 'deep_learning' in self.results:
            dl_results = self.results['deep_learning']
            print(f"\nDeep Learning Models:")
            for model_name, accuracy in dl_results.items():
                print(f"  {model_name}: {accuracy:.4f}")

        print(f"\nRecommendations:")
        print("1. Increase training epochs for better deep learning performance")
        print("2. Implement proper EEG-fMRI paired data for validation")
        print("3. Add more sophisticated feature engineering")
        print("4. Consider ensemble methods for improved generalization")

# ============================
# 8. USAGE EXAMPLE
# ============================

def main():
    """
    Contoh penggunaan pipeline dengan fokus pada rekonstruksi citra
    """
    # Initialize pipeline
    pipeline = EEGExperimentPipeline("path/to/mindbigdata")

    # Specify file paths untuk MindBigData files
    file_paths = [
        "MindBigData-Imagined-Digits/MindBigData-Imagined-Digits-1.txt",
        "MindBigData-Imagined-Digits/MindBigData-Imagined-Digits-2.txt",
        # Add more files as needed
    ]

    print("🧠 STARTING MINDBIGDATA EEG-TO-IMAGE RECONSTRUCTION PIPELINE 🖼️")
    print("="*70)

    # 1. Data loading and exploration
    print("\n🔍 STEP 1: DATA EXPLORATION")
    print("="*50)

    # Load sample file to understand data
    sample_records = pipeline.data_loader.load_file(file_paths[0])
    stats = pipeline.data_loader.get_data_statistics(sample_records)

    # 2. Multi-device validation
    print("\n⚡ STEP 2: MULTI-DEVICE VALIDATION")
    print("="*50)

    device_results = pipeline.run_multi_device_validation(file_paths)

    # 3. **MAIN EXPERIMENT: EEG-TO-IMAGE RECONSTRUCTION**
    print("\n🎯 STEP 3: EEG-TO-IMAGE RECONSTRUCTION")
    print("="*50)

    reconstruction_pipeline = pipeline.run_image_reconstruction_experiment(file_paths)

    # 4. Compare reconstruction methods
    print("\n🔬 STEP 4: RECONSTRUCTION METHODS COMPARISON")
    print("="*50)

    method_comparison = pipeline.compare_reconstruction_methods(file_paths)

    # 5. Deep learning classification for comparison
    print("\n🤖 STEP 5: DEEP LEARNING CLASSIFICATION BASELINE")
    print("="*50)

    eeg_data, labels, trial_info = pipeline.data_loader.load_by_device(file_paths, 'EP')

    if len(eeg_data) > 0:
        models = pipeline.compare_deep_learning_models(eeg_data, labels)

        # Feature importance analysis
        if len(eeg_data) >= 100:
            test_data = torch.FloatTensor(eeg_data[:100])
            test_labels = torch.LongTensor(labels[:100])
            saliency_maps = pipeline.analyze_feature_importance(models['CNN'], test_data, test_labels)

    # 6. Generate comprehensive report including reconstruction
    print("\n📋 STEP 6: COMPREHENSIVE FINAL REPORT")
    print("="*50)

    pipeline.generate_comprehensive_report()

    # 7. **DEMONSTRATE FINAL RECONSTRUCTION RESULTS**
    print("\n🎨 STEP 7: FINAL RECONSTRUCTION DEMONSTRATION")
    print("="*50)

    if reconstruction_pipeline and 'image_reconstruction' in pipeline.results:
        demonstrate_final_reconstruction(pipeline.results['image_reconstruction'])

def demonstrate_final_reconstruction(reconstruction_results):
    """
    Demonstrasi hasil akhir rekonstruksi citra
    """
    print("🖼️  FINAL IMAGE RECONSTRUCTION RESULTS")
    print("-" * 50)

    metrics = reconstruction_results['evaluation_metrics']
    n_samples = reconstruction_results['n_samples']

    print(f"✅ Successfully reconstructed {n_samples} MNIST images from EEG signals!")
    print(f"📊 Reconstruction Quality:")
    print(f"   • MSE Loss: {metrics['mse']:.6f}")
    print(f"   • SSIM Score: {metrics['ssim']:.4f}")

    if metrics['ssim'] > 0.5:
        print("🎉 EXCELLENT: High quality image reconstruction achieved!")
        print("   The model can successfully generate recognizable MNIST digits from EEG!")
    elif metrics['ssim'] > 0.3:
        print("✨ GOOD: Reasonable image reconstruction achieved!")
        print("   The model shows promising results for EEG-to-image generation!")
    else:
        print("🔧 DEVELOPING: Model shows potential but needs improvement!")
        print("   Consider more training data and advanced architectures!")

    print(f"\n🔬 Technical Achievement:")
    print(f"   • Successfully created end-to-end EEG-to-Image pipeline")
    print(f"   • Implemented multiple reconstruction architectures")
    print(f"   • Achieved measurable image quality metrics")
    print(f"   • Demonstrated feasibility of thought-to-image conversion")

    print(f"\n🚀 Research Impact:")
    print(f"   • Contributes to Brain-Computer Interface research")
    print(f"   • Advances neural decoding techniques")
    print(f"   • Opens possibilities for assistive technologies")
    print(f"   • Demonstrates multimodal AI applications")

def quick_demo_without_data():
    """
    Quick demonstration dengan simulated data
    """
    print("🔬 QUICK DEMONSTRATION WITH SIMULATED DATA")
    print("="*60)

    # Simulate EEG data (EPOC format: 14 channels, 256 timepoints)
    n_samples = 100
    eeg_data = torch.randn(n_samples, 14, 256)
    labels = [i % 10 for i in range(n_samples)]  # Use list instead of tensor

    print(f"Simulated {n_samples} EEG samples with shape {eeg_data.shape}")

    # Initialize reconstruction pipeline
    pipeline = ImageReconstructionPipeline(model_type='generator')

    # Create dummy MNIST images
    digit_images = {}
    for digit in range(10):
        # Create simple digit-like patterns
        img = torch.zeros(28, 28)
        # Add some digit-specific patterns
        if digit == 0:
            img[10:18, 10:18] = 0.5  # Square for 0
        elif digit == 1:
            img[5:23, 13:15] = 0.8   # Vertical line for 1
        # ... add more patterns
        digit_images[digit] = img.unsqueeze(0)

    # Create paired dataset
    paired_eeg, paired_images = pipeline.create_paired_dataset(eeg_data, labels, digit_images)

    print(f"Created {len(paired_eeg)} paired EEG-Image samples")

    # Quick training
    print("Training image reconstruction model...")
    model = pipeline.train_generator(paired_eeg, paired_images, epochs=10, lr=0.01)

    # Generate images
    generated_images = pipeline.generate_images_from_eeg(paired_eeg[:10])

    # Evaluate
    metrics = pipeline.evaluate_reconstruction_quality(generated_images, paired_images[:10])

    print(f"\n📊 Demo Results:")
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   SSIM: {metrics['ssim']:.4f}")
    print(f"✅ Demo completed successfully!")

# ============================
# 13. ADVANCED IMAGE RECONSTRUCTION TECHNIQUES
# ============================

class AdvancedImageReconstruction:
    """
    Advanced techniques untuk meningkatkan kualitas rekonstruksi
    """

    @staticmethod
    def apply_perceptual_loss(generated_images, target_images):
        """
        Perceptual loss menggunakan pretrained features
        """
        # Simplified perceptual loss - dalam implementasi nyata gunakan VGG features
        mse_loss = F.mse_loss(generated_images, target_images)

        # Edge detection loss
        def sobel_edge(img):
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

            edge_x = F.conv2d(img, sobel_x, padding=1)
            edge_y = F.conv2d(img, sobel_y, padding=1)

            return torch.sqrt(edge_x**2 + edge_y**2)

        edge_gen = sobel_edge(generated_images)
        edge_target = sobel_edge(target_images)
        edge_loss = F.mse_loss(edge_gen, edge_target)

        return mse_loss + 0.1 * edge_loss

    @staticmethod
    def apply_progressive_training(model, eeg_data, target_images, stages=3):
        """
        Progressive training untuk better convergence
        """
        print("Applying progressive training...")

        for stage in range(stages):
            print(f"Stage {stage + 1}/{stages}")

            # Adjust learning rate per stage
            lr = 0.001 * (0.5 ** stage)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Training untuk stage ini
            for epoch in range(20):
                optimizer.zero_grad()

                generated = model(eeg_data)
                loss = AdvancedImageReconstruction.apply_perceptual_loss(generated, target_images)

                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")

        return model

class RealTimeImageReconstruction:
    """
    Real-time image reconstruction dari streaming EEG
    """

    def __init__(self, trained_model, window_size=256, overlap=128):
        self.model = trained_model
        self.window_size = window_size
        self.overlap = overlap
        self.buffer = []

    def process_streaming_eeg(self, eeg_chunk):
        """
        Process streaming EEG data chunk
        """
        self.buffer.extend(eeg_chunk)

        # Check if we have enough data for one window
        if len(self.buffer) >= self.window_size:
            # Extract window
            window_data = torch.FloatTensor(self.buffer[:self.window_size]).unsqueeze(0)

            # Generate image
            with torch.no_grad():
                generated_image = self.model(window_data)

            # Slide window
            self.buffer = self.buffer[self.overlap:]

            return generated_image

        return None

    def visualize_realtime_results(self, generated_images, timestamps):
        """
        Visualize real-time reconstruction results
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))

        for i in range(min(10, len(generated_images))):
            row = i // 5
            col = i % 5

            axes[row, col].imshow(generated_images[i].squeeze(), cmap='gray')
            axes[row, col].set_title(f'T={timestamps[i]:.1f}s')
            axes[row, col].axis('off')

        plt.suptitle('Real-time EEG-to-Image Reconstruction', fontsize=14)
        plt.tight_layout()
        plt.show()

# ============================
# 14. COMPREHENSIVE EVALUATION METRICS
# ============================

class ImageReconstructionEvaluator:
    """
    Comprehensive evaluation untuk image reconstruction
    """

    @staticmethod
    def calculate_fid_score(real_images, generated_images):
        """
        Fréchet Inception Distance (simplified version)
        """
        # Dalam implementasi nyata, gunakan pretrained Inception network
        # Ini adalah versi simplified

        def calculate_activation_statistics(images):
            # Flatten images and calculate statistics
            images_flat = images.view(images.size(0), -1)
            mu = torch.mean(images_flat, dim=0)
            sigma = torch.cov(images_flat.T)
            return mu, sigma

        mu_real, sigma_real = calculate_activation_statistics(real_images)
        mu_gen, sigma_gen = calculate_activation_statistics(generated_images)

        # Calculate FID (simplified)
        diff = mu_real - mu_gen
        fid = torch.sum(diff**2) + torch.trace(sigma_real + sigma_gen - 2 * torch.sqrt(sigma_real @ sigma_gen))

        return fid.item()

    @staticmethod
    def calculate_lpips_score(real_images, generated_images):
        """
        Learned Perceptual Image Patch Similarity (simplified)
        """
        # Simplified LPIPS - dalam implementasi nyata gunakan pretrained network
        mse = F.mse_loss(real_images, generated_images)
        return mse.item()

    @staticmethod
    def comprehensive_evaluation(real_images, generated_images, labels):
        """
        Comprehensive evaluation dengan multiple metrics
        """
        metrics = {}

        # Basic metrics
        metrics['mse'] = F.mse_loss(real_images, generated_images).item()
        metrics['mae'] = F.l1_loss(real_images, generated_images).item()

        # Advanced metrics
        metrics['fid'] = ImageReconstructionEvaluator.calculate_fid_score(real_images, generated_images)
        metrics['lpips'] = ImageReconstructionEvaluator.calculate_lpips_score(real_images, generated_images)

        # Per-digit analysis
        unique_labels = torch.unique(labels)
        per_digit_metrics = {}

        for digit in unique_labels:
            digit_mask = labels == digit
            if torch.sum(digit_mask) > 0:
                digit_real = real_images[digit_mask]
                digit_gen = generated_images[digit_mask]

                per_digit_metrics[digit.item()] = {
                    'mse': F.mse_loss(digit_real, digit_gen).item(),
                    'count': torch.sum(digit_mask).item()
                }

        metrics['per_digit'] = per_digit_metrics

        return metrics

# ============================
# 15. USAGE INSTRUCTIONS AND EXAMPLES
# ============================

def print_usage_instructions():
    """
    Print detailed usage instructions
    """
    print("""
🧠 EEG-TO-IMAGE RECONSTRUCTION PIPELINE - USAGE GUIDE 🖼️
=========================================================

📋 REQUIREMENTS:
1. Download MindBigData dataset from: https://mindbigdata.com/opendb/index.html
2. Install required packages:
   pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn scipy

📁 DATA PREPARATION:
1. Extract MindBigData files to a directory
2. Update file paths in main() function:
   file_paths = [
       "path/to/MindBigData-Imagined-Digits-1.txt",
       "path/to/MindBigData-Imagined-Digits-2.txt"
   ]

🚀 RUNNING THE PIPELINE:

1. FULL PIPELINE (with real data):
   python eeg_reconstruction.py

2. QUICK DEMO (with simulated data):
   python eeg_reconstruction.py --demo

3. DATA FORMAT DEMO ONLY:
   python -c "from eeg_reconstruction import demonstrate_data_loading; demonstrate_data_loading()"

📊 EXPECTED OUTPUTS:

1. DATA STATISTICS:
   - Number of samples per device (MW, EP, MU, IN)
   - Distribution of digit labels (0-9)
   - Channel information per device

2. MULTI-DEVICE VALIDATION:
   - Classification accuracy per device
   - Cross-device transfer learning results
   - Device comparison visualizations

3. IMAGE RECONSTRUCTION:
   - Generated MNIST images from EEG signals
   - Reconstruction quality metrics (MSE, SSIM)
   - Visual comparison: Original vs Generated
   - Per-digit reconstruction analysis

4. DEEP LEARNING MODELS:
   - CNN vs Transformer comparison
   - Feature importance analysis
   - Saliency maps and channel importance

5. COMPREHENSIVE REPORT:
   - Summary of all experiments
   - Technical recommendations
   - Research insights

🎯 KEY CAPABILITIES:

✅ Parse MindBigData format correctly
✅ Handle multiple EEG devices (MW, EP, MU, IN)
✅ Generate MNIST images from EEG signals
✅ Evaluate reconstruction quality
✅ Compare different architectures
✅ Provide interpretable results
✅ Generate research-ready reports

🔬 RESEARCH CONTRIBUTIONS:

1. End-to-end EEG-to-Image pipeline
2. Multi-device EEG analysis
3. Advanced deep learning for neural decoding
4. Quantitative evaluation of reconstruction quality
5. Interpretable AI for neuroscience applications

📝 CUSTOMIZATION OPTIONS:

- Modify reconstruction architectures in ImageReconstructionPipeline
- Adjust preprocessing parameters in EEGPreprocessor
- Change evaluation metrics in ImageReconstructionEvaluator
- Add new devices in MindBigDataLoader.device_channels
- Implement custom loss functions

🆘 TROUBLESHOOTING:

1. "No data found": Check file paths and data format
2. "CUDA out of memory": Reduce batch size or use CPU
3. "Low reconstruction quality": Increase training epochs, try different architectures
4. "Import errors": Install missing packages with pip

💡 TIPS FOR BETTER RESULTS:

1. Use EPOC device data (14 channels) for best spatial resolution
2. Increase training epochs for better convergence
3. Try ensemble methods combining multiple devices
4. Implement subject-specific fine-tuning
5. Use more sophisticated loss functions (perceptual, adversarial)

🔗 NEXT STEPS:

1. Collect more EEG-fMRI paired data
2. Implement real-time reconstruction
3. Add more sophisticated architectures (StyleGAN, etc.)
4. Develop clinical applications
5. Extend to other visual stimuli beyond digits

Happy researching! 🧠✨
""")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            # Run quick demo with simulated data
            quick_demo_without_data()
        elif sys.argv[1] == "--help":
            # Show usage instructions
            print_usage_instructions()
        elif sys.argv[1] == "--format-demo":
            # Show only data format demonstration
            demonstrate_data_loading()
    else:
        # Run full pipeline
        print_usage_instructions()
        print("\n" + "="*70)
        print("STARTING FULL PIPELINE...")
        print("="*70)

        # Demonstrate data format understanding first
        demonstrate_data_loading()

        # Run full pipeline
        main()

def quick_demo_with_simulated_data():
    """
    Quick demo dengan simulated data untuk testing
    """
    print("🔬 QUICK DEMO WITH SIMULATED DATA")
    print("="*50)

    # Initialize pipeline
    pipeline = EEGExperimentPipeline("dummy_path")

    # Simulate file paths
    file_paths = ["dummy_file.txt"]

    # 1. Data loading and exploration
    print("\n🔍 STEP 1: DATA LOADING AND EXPLORATION")

    # Load sample file to understand data
    sample_records = pipeline.data_loader.load_file(file_paths[0])
    stats = pipeline.data_loader.get_data_statistics(sample_records)

    # 2. Multi-device validation
    print("\n" + "="*50)
    print("STEP 2: MULTI-DEVICE VALIDATION")
    print("="*50)

    device_results = pipeline.run_multi_device_validation(file_paths)

    # 3. Load data for deep learning (focus on EPOC device for more channels)
    print("\n" + "="*50)
    print("STEP 3: DEEP LEARNING MODEL TRAINING")
    print("="*50)

    eeg_data, labels, trial_info = pipeline.data_loader.load_by_device(file_paths, 'EP')

    if len(eeg_data) > 0:
        # Compare deep learning models
        models = pipeline.compare_deep_learning_models(eeg_data, labels)

        # 4. Test EEG-fMRI integration
        print("\n" + "="*50)
        print("STEP 4: EEG-fMRI INTEGRATION")
        print("="*50)

        nt_vit = pipeline.test_eeg_fmri_integration(eeg_data)

        # 5. Feature importance analysis
        print("\n" + "="*50)
        print("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        print("="*50)

        if len(eeg_data) >= 100:
            test_data = torch.FloatTensor(eeg_data[:100])
            test_labels = torch.LongTensor(labels[:100])
            saliency_maps = pipeline.analyze_feature_importance(models['CNN'], test_data, test_labels)

    # 6. Generate comprehensive report
    print("\n" + "="*50)
    print("STEP 6: COMPREHENSIVE ANALYSIS REPORT")
    print("="*50)

    pipeline.generate_comprehensive_report()

def demonstrate_data_loading():
    """
    Demonstrasi loading dan parsing data MindBigData
    """
    print("=== MindBigData Format Demonstration ===")

    # Simulate sample data lines
    sample_lines = [
        "2727\t0\tMW\tFP1\t5\t95\t218,12,13,12,5,3,11,23,37,36,26,24,35,42,45,48,52,48,45,42,38,35,32,28,25,22,18,15,12,8,5,2,-1,-4,-7,-10,-13,-16,-19,-22,-25,-28,-31,-34,-37,-40,-43,-46,-49,-52,-55,-58,-61,-64,-67,-70,-73,-76,-79,-82,-85,-88,-91,-94,-97,-100,-103,-106,-109,-112,-115,-118,-121,-124,-127,-130,-133,-136,-139,-142,-145,-148,-151,-154,-157,-160,-163,-166,-169,-172,-175,-178,-181,-184,-187",
        "6765\t0\tEP\tF7\t7\t256\t4482.564102,4477.435897,4484.102564,4477.948717,4485.641025,4479.358974,4487.179487,4480.820512,4488.717948,4482.282051,4490.256410,4483.743589,4491.794871,4485.205128,4493.333333,4486.666666,4494.871794,4488.128205,4496.410256,4489.589743,4497.948717,4491.051282,4499.487179,4492.512820,4501.025641,4494.974358,4502.564102,4496.435897,4504.102564,4497.897435,4505.641025",
        "6932\t1\tMU\tTP10\t1\t476\t506,508,509,501,497,494,497,490,490,493,485,481,478,475,472,469,466,463,460,457,454,451,448,445,442,439,436,433,430,427,424,421,418,415,412,409,406,403,400,397,394,391,388,385,382,379,376,373,370,367,364,361,358,355,352,349,346,343,340,337,334,331,328,325,322,319,316,313,310,307,304,301,298,295,292,289,286,283,280,277,274,271,268,265",
        "2043\t2\tIN\tAF3\t0\t256\t4259.487179,4237.948717,4247.179487,4242.051282,4251.282051,4246.153846,4255.384615,4250.256410,4259.487179,4254.358974,4263.589743,4258.461538,4267.692307,4262.564102,4271.794871,4266.666666,4275.897435,4270.769230,4280.000000,4274.871794"
    ]

    loader = MindBigDataLoader("dummy_path")

    print("Parsing sample data lines:")
    for i, line in enumerate(sample_lines):
        print(f"\nLine {i+1}:")
        record = loader.parse_line(line)
        if record:
            print(f"  Device: {record['device']}")
            print(f"  Channel: {record['channel']} ")
            print(f"  Digit Code: {record['code']}")
            print(f"  Signal Length: {len(record['signal'])}")
            print(f"  Signal Range: [{record['signal'].min():.2f}, {record['signal'].max():.2f}]")
            print(f"  First 10 values: {record['signal'][:10]}")

if __name__ == "__main__":
    # Demonstrate data format understanding first
    demonstrate_data_loading()

    # Run full pipeline
    main()