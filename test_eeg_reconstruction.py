"""
Comprehensive test suite for EEG reconstruction pipeline
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import the classes we want to test
from eeg_reconstruction import (
    MindBigDataLoader,
    EEGPreprocessor,
    EEG_CNN,
    EEGTransformer,
    EEGToImageGenerator,
    ImageReconstructionPipeline,
    EEGExperimentPipeline,
    EEGVisualizationTools,
    EEGStatisticalAnalysis
)


class TestMindBigDataLoader:
    """Test the MindBigData loader functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.loader = MindBigDataLoader("dummy_path")

    def test_initialization(self):
        """Test loader initialization"""
        assert self.loader.data_path == "dummy_path"
        assert 'MW' in self.loader.device_channels
        assert 'EP' in self.loader.device_channels
        assert len(self.loader.device_channels['EP']) == 14  # EPOC has 14 channels

    def test_parse_line_valid(self):
        """Test parsing valid data line"""
        # Sample line from MindBigData format
        line = "1\t2\tEP\tAF3\t5\t10\t1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.0"

        record = self.loader.parse_line(line)

        assert record is not None
        assert record['id'] == 1
        assert record['event'] == 2
        assert record['device'] == 'EP'
        assert record['channel'] == 'AF3'
        assert record['code'] == 5
        assert record['size'] == 10
        assert len(record['signal']) == 10
        assert record['signal'][0] == 1.1

    def test_parse_line_invalid(self):
        """Test parsing invalid data line"""
        invalid_line = "invalid\tdata\tformat"
        record = self.loader.parse_line(invalid_line)
        assert record is None

    def test_organize_by_trials(self):
        """Test trial organization"""
        # Create mock records
        records = [
            {'event': 1, 'code': 5, 'device': 'EP', 'channel': 'AF3',
             'signal': np.array([1, 2, 3]), 'size': 3, 'id': 1},
            {'event': 1, 'code': 5, 'device': 'EP', 'channel': 'F3',
             'signal': np.array([4, 5, 6]), 'size': 3, 'id': 2}
        ]

        trials = self.loader.organize_by_trials(records)

        assert len(trials) == 1  # One trial
        trial_key = "1_5_EP"
        assert trial_key in trials
        assert len(trials[trial_key]['channels']) == 2  # Two channels


class TestEEGPreprocessor:
    """Test EEG preprocessing functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = EEGPreprocessor(sampling_rate=250)

    def test_initialization(self):
        """Test preprocessor initialization"""
        assert self.preprocessor.sampling_rate == 250
        assert self.preprocessor.scaler is not None

    def test_extract_features(self):
        """Test feature extraction"""
        # Create mock EEG data: (n_samples, n_channels, n_timepoints)
        eeg_data = np.random.randn(10, 14, 256)

        features = self.preprocessor.extract_features(eeg_data)

        assert features.shape[0] == 10  # Same number of samples
        assert features.shape[1] > 0   # Features extracted
        assert not np.isnan(features).any()  # No NaN values

    @patch('scipy.signal.butter')
    @patch('scipy.signal.filtfilt')
    def test_bandpass_filter(self, mock_filtfilt, mock_butter):
        """Test bandpass filtering"""
        # Mock scipy functions
        mock_butter.return_value = ([1, 2, 3], [4, 5, 6])
        mock_filtfilt.return_value = np.random.randn(256)

        eeg_data = np.random.randn(5, 14, 256)
        filtered_data = self.preprocessor.bandpass_filter(eeg_data)

        assert filtered_data.shape == eeg_data.shape
        assert mock_butter.called
        assert mock_filtfilt.called


class TestDeepLearningModels:
    """Test deep learning model architectures"""

    def test_eeg_cnn_forward(self):
        """Test EEG CNN forward pass"""
        model = EEG_CNN(n_channels=14, n_timepoints=256, n_classes=10)

        # Create mock input
        x = torch.randn(8, 14, 256)  # batch_size=8

        output = model(x)

        assert output.shape == (8, 10)  # batch_size x n_classes
        assert not torch.isnan(output).any()

    def test_eeg_transformer_forward(self):
        """Test EEG Transformer forward pass"""
        model = EEGTransformer(n_channels=14, n_timepoints=256, n_classes=10)

        # Create mock input
        x = torch.randn(4, 14, 256)  # batch_size=4

        output = model(x)

        assert output.shape == (4, 10)  # batch_size x n_classes
        assert not torch.isnan(output).any()

    def test_eeg_to_image_generator(self):
        """Test EEG to Image Generator"""
        model = EEGToImageGenerator(eeg_channels=14, eeg_timepoints=256, image_size=28)

        # Create mock EEG input
        x = torch.randn(4, 14, 256)

        output = model(x)

        assert output.shape == (4, 1, 28, 28)  # batch_size x channels x height x width
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


class TestImageReconstructionPipeline:
    """Test image reconstruction pipeline"""

    def setup_method(self):
        """Setup test fixtures"""
        self.pipeline = ImageReconstructionPipeline(model_type='generator')

    def test_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.model_type == 'generator'
        assert self.pipeline.model is None
        assert isinstance(self.pipeline.criterion, nn.MSELoss)

    @patch('torchvision.datasets.MNIST')
    def test_load_mnist_references(self, mock_mnist):
        """Test MNIST reference loading"""
        # Mock MNIST dataset
        mock_dataset = []
        for digit in range(10):
            mock_dataset.append((torch.randn(1, 28, 28), digit))

        mock_mnist.return_value = mock_dataset

        digit_images = self.pipeline.load_mnist_references()

        assert len(digit_images) == 10
        for digit in range(10):
            assert digit in digit_images

    def test_create_paired_dataset(self):
        """Test paired dataset creation"""
        # Create mock data with specific labels that exist in digit_images
        eeg_data = torch.randn(20, 14, 256)
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2  # Use list instead of tensor

        digit_images = {}
        for digit in range(10):
            digit_images[digit] = torch.randn(1, 28, 28)

        paired_eeg, paired_images = self.pipeline.create_paired_dataset(
            eeg_data, labels, digit_images
        )

        assert len(paired_eeg) == len(paired_images)
        assert len(paired_eeg) == 20  # All labels should have corresponding images

    def test_evaluate_reconstruction_quality(self):
        """Test reconstruction quality evaluation"""
        # Create mock images
        generated = torch.randn(5, 1, 28, 28)
        target = torch.randn(5, 1, 28, 28)

        metrics = self.pipeline.evaluate_reconstruction_quality(generated, target)

        assert 'mse' in metrics
        assert 'ssim' in metrics
        assert 'individual_ssim' in metrics
        assert len(metrics['individual_ssim']) == 5


class TestEEGExperimentPipeline:
    """Test the main experiment pipeline"""

    def setup_method(self):
        """Setup test fixtures"""
        self.pipeline = EEGExperimentPipeline("dummy_path")

    def test_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.data_loader is not None
        assert self.pipeline.preprocessor is not None
        assert isinstance(self.pipeline.results, dict)

    @patch.object(MindBigDataLoader, 'load_by_device')
    def test_compare_deep_learning_models(self, mock_load):
        """Test deep learning model comparison"""
        # Mock data loading with correct dimensions
        mock_eeg_data = np.random.randn(50, 14, 200)  # Reduced timepoints to avoid dimension issues
        mock_labels = np.random.randint(0, 10, 50)
        mock_load.return_value = (mock_eeg_data, mock_labels, [])

        models = self.pipeline.compare_deep_learning_models(mock_eeg_data, mock_labels)

        assert 'CNN' in models
        assert 'Transformer' in models
        assert 'deep_learning' in self.pipeline.results


class TestVisualizationTools:
    """Test visualization functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.viz_tools = EEGVisualizationTools()

    def test_initialization(self):
        """Test visualization tools initialization"""
        assert self.viz_tools is not None

    @patch('matplotlib.pyplot.show')
    def test_plot_device_comparison(self, mock_show):
        """Test device comparison plotting"""
        device_results = {
            'MW': {'accuracy': 0.75, 'n_samples': 100, 'n_channels': 1, 'n_features': 100},
            'EP': {'accuracy': 0.85, 'n_samples': 200, 'n_channels': 14, 'n_features': 1400},
            'MU': {'accuracy': 0.70, 'n_samples': 150, 'n_channels': 4, 'n_features': 400},
            'IN': {'accuracy': 0.80, 'n_samples': 180, 'n_channels': 5, 'n_features': 500}
        }

        # Should not raise an exception
        self.viz_tools.plot_device_comparison(device_results)
        assert mock_show.called


class TestStatisticalAnalysis:
    """Test statistical analysis functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.stats_analyzer = EEGStatisticalAnalysis()

    def test_initialization(self):
        """Test statistical analyzer initialization"""
        assert self.stats_analyzer is not None

    def test_compute_effect_sizes(self):
        """Test effect size computation"""
        # Create mock EEG data
        eeg_data = np.random.randn(100, 14, 256)
        labels = np.random.randint(0, 5, 100)  # 5 different labels

        # Should not raise an exception
        self.stats_analyzer.compute_effect_sizes(eeg_data, labels)


# Integration tests
class TestIntegration:
    """Integration tests for the full pipeline"""

    @patch.object(MindBigDataLoader, 'load_file')
    def test_full_pipeline_with_mock_data(self, mock_load_file):
        """Test the full pipeline with mocked data"""
        # Mock file loading
        mock_records = [
            {'event': 1, 'code': 5, 'device': 'EP', 'channel': 'AF3',
             'signal': np.random.randn(256), 'size': 256, 'id': 1}
        ]
        mock_load_file.return_value = mock_records

        pipeline = EEGExperimentPipeline("dummy_path")

        # Test data statistics
        stats = pipeline.data_loader.get_data_statistics(mock_records)
        assert 'devices' in stats
        assert 'codes' in stats
        assert 'channels' in stats


# Performance tests
class TestPerformance:
    """Performance and memory tests"""

    def test_memory_usage_large_dataset(self):
        """Test memory usage with large dataset"""
        # Create large mock dataset
        large_eeg_data = np.random.randn(1000, 14, 256)

        preprocessor = EEGPreprocessor()
        features = preprocessor.extract_features(large_eeg_data)

        # Should complete without memory errors
        assert features.shape[0] == 1000

    def test_model_inference_speed(self):
        """Test model inference speed"""
        model = EEG_CNN(n_channels=14, n_timepoints=256, n_classes=10)
        model.eval()

        # Test batch inference
        x = torch.randn(32, 14, 256)

        import time
        start_time = time.time()

        with torch.no_grad():
            output = model(x)

        inference_time = time.time() - start_time

        # Should complete inference quickly (less than 1 second for this small model)
        assert inference_time < 1.0
        assert output.shape == (32, 10)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
