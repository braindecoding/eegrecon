"""
Pytest configuration and shared fixtures for EEG reconstruction tests
"""

import pytest
import numpy as np
import torch
import tempfile
import os
from unittest.mock import MagicMock


@pytest.fixture
def sample_eeg_data():
    """Generate sample EEG data for testing"""
    # Shape: (n_samples, n_channels, n_timepoints)
    return np.random.randn(20, 14, 256)


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing"""
    return np.random.randint(0, 10, 20)


@pytest.fixture
def sample_torch_eeg():
    """Generate sample EEG data as PyTorch tensor"""
    return torch.randn(10, 14, 256)


@pytest.fixture
def sample_mnist_images():
    """Generate sample MNIST-like images"""
    digit_images = {}
    for digit in range(10):
        digit_images[digit] = torch.randn(1, 28, 28)
    return digit_images


@pytest.fixture
def mock_mindbigdata_line():
    """Sample MindBigData format line"""
    return "1\t2\tEP\tAF3\t5\t10\t1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.0"


@pytest.fixture
def mock_mindbigdata_records():
    """Sample MindBigData records"""
    return [
        {
            'id': 1, 'event': 1, 'device': 'EP', 'channel': 'AF3',
            'code': 5, 'size': 256, 'signal': np.random.randn(256)
        },
        {
            'id': 2, 'event': 1, 'device': 'EP', 'channel': 'F3',
            'code': 5, 'size': 256, 'signal': np.random.randn(256)
        },
        {
            'id': 3, 'event': 2, 'device': 'EP', 'channel': 'AF3',
            'code': 7, 'size': 256, 'signal': np.random.randn(256)
        }
    ]


@pytest.fixture
def temp_data_file():
    """Create a temporary data file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        # Write sample MindBigData format lines
        f.write("1\t1\tEP\tAF3\t5\t10\t1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.0\n")
        f.write("2\t1\tEP\tF3\t5\t10\t2.1,3.2,4.3,5.4,6.5,7.6,8.7,9.8,10.9,11.0\n")
        f.write("3\t2\tEP\tAF3\t7\t10\t3.1,4.2,5.3,6.4,7.5,8.6,9.7,10.8,11.9,12.0\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def device_results_sample():
    """Sample device results for testing visualization"""
    return {
        'MW': {
            'accuracy': 0.75,
            'n_samples': 100,
            'n_channels': 1,
            'n_features': 150
        },
        'EP': {
            'accuracy': 0.85,
            'n_samples': 200,
            'n_channels': 14,
            'n_features': 1400
        },
        'MU': {
            'accuracy': 0.70,
            'n_samples': 150,
            'n_channels': 4,
            'n_features': 400
        },
        'IN': {
            'accuracy': 0.80,
            'n_samples': 180,
            'n_channels': 5,
            'n_features': 500
        }
    }


@pytest.fixture
def cross_device_results_sample():
    """Sample cross-device results for testing"""
    return {
        'MW_to_EP': 0.65,
        'MW_to_MU': 0.60,
        'MW_to_IN': 0.62,
        'EP_to_MW': 0.55,
        'EP_to_MU': 0.70,
        'EP_to_IN': 0.75,
        'MU_to_MW': 0.50,
        'MU_to_EP': 0.68,
        'MU_to_IN': 0.65,
        'IN_to_MW': 0.52,
        'IN_to_EP': 0.72,
        'IN_to_MU': 0.63
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests"""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to prevent actual plotting during tests"""
    import matplotlib.pyplot as plt
    original_show = plt.show
    plt.show = MagicMock()
    yield plt
    plt.show = original_show


# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["large", "performance", "slow"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark unit tests (default)
        if not any(marker.name in ["integration", "slow"] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
