#!/usr/bin/env python3
"""
Test runner script for EEG reconstruction project
"""

import subprocess
import sys
import os
import time


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")

    required_packages = [
        'numpy', 'pandas', 'torch', 'torchvision',
        'matplotlib', 'seaborn', 'sklearn', 'scipy'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - MISSING")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print("pip install -r requirements.txt")
        return False

    print("✅ All dependencies are installed!")
    return True


def run_quick_demo():
    """Run quick functionality demo"""
    print("\n⚡ Running quick functionality demo...")

    try:
        result = subprocess.run([
            sys.executable, "demo_script.py", "--quick"
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ Quick demo passed!")
            print(result.stdout)
            return True
        else:
            print("❌ Quick demo failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Quick demo timed out!")
        return False
    except Exception as e:
        print(f"❌ Quick demo error: {e}")
        return False


def run_unit_tests():
    """Run unit tests with pytest"""
    print("\n🧪 Running unit tests...")

    if not os.path.exists("test_eeg_reconstruction.py"):
        print("❌ Test file not found!")
        return False

    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "test_eeg_reconstruction.py",
            "-v", "--tb=short", "-x"  # Stop on first failure
        ], capture_output=True, text=True, timeout=300)

        print("PYTEST OUTPUT:")
        print(result.stdout)

        if result.stderr:
            print("PYTEST ERRORS:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ All unit tests passed!")
            return True
        else:
            print("❌ Some unit tests failed!")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Unit tests timed out!")
        return False
    except Exception as e:
        print(f"❌ Unit test error: {e}")
        return False


def run_comprehensive_demo():
    """Run comprehensive demo"""
    print("\n🚀 Running comprehensive demo...")

    try:
        result = subprocess.run([
            sys.executable, "demo_script.py"
        ], capture_output=True, text=True, timeout=600)

        print("DEMO OUTPUT:")
        print(result.stdout)

        if result.stderr:
            print("DEMO ERRORS:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ Comprehensive demo completed!")
            return True
        else:
            print("❌ Comprehensive demo failed!")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Comprehensive demo timed out!")
        return False
    except Exception as e:
        print(f"❌ Comprehensive demo error: {e}")
        return False


def run_import_test():
    """Test if the main module can be imported"""
    print("\n📦 Testing module import...")

    try:
        import eeg_reconstruction
        print("✅ Successfully imported eeg_reconstruction module!")

        # Test key classes
        classes_to_test = [
            'MindBigDataLoader',
            'EEGPreprocessor',
            'EEG_CNN',
            'EEGTransformer',
            'EEGToImageGenerator',
            'ImageReconstructionPipeline'
        ]

        for class_name in classes_to_test:
            if hasattr(eeg_reconstruction, class_name):
                print(f"   ✅ {class_name}")
            else:
                print(f"   ❌ {class_name} - NOT FOUND")
                return False

        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def run_comprehensive_validation():
    """Run comprehensive validation of the entire system"""
    print("\n🔬 Running comprehensive system validation...")

    try:
        # Test all major components
        import numpy as np
        import torch
        from eeg_reconstruction import (
            MindBigDataLoader, EEGPreprocessor, EEG_CNN,
            EEGTransformer, EEGToImageGenerator, ImageReconstructionPipeline
        )

        print("   ✅ All major classes imported successfully")

        # Test data loading
        loader = MindBigDataLoader("dummy")
        sample_line = "1\t2\tEP\tAF3\t5\t10\t1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10.0"
        record = loader.parse_line(sample_line)
        assert record is not None
        print("   ✅ Data parsing works correctly")

        # Test preprocessing
        preprocessor = EEGPreprocessor()
        mock_data = np.random.randn(10, 14, 256)
        features = preprocessor.extract_features(mock_data)
        assert features.shape[0] == 10
        print("   ✅ EEG preprocessing works correctly")

        # Test models
        cnn = EEG_CNN(n_channels=14, n_timepoints=256, n_classes=10)
        transformer = EEGTransformer(n_channels=14, n_timepoints=256, n_classes=10)
        generator = EEGToImageGenerator(eeg_channels=14, eeg_timepoints=256)

        test_input = torch.randn(5, 14, 256)
        cnn_out = cnn(test_input)
        trans_out = transformer(test_input)
        gen_out = generator(test_input)

        assert cnn_out.shape == (5, 10)
        assert trans_out.shape == (5, 10)
        assert gen_out.shape == (5, 1, 28, 28)
        print("   ✅ All deep learning models work correctly")

        # Test image reconstruction pipeline
        pipeline = ImageReconstructionPipeline()
        labels = [0, 1, 2, 3, 4]
        digit_images = {i: torch.randn(1, 28, 28) for i in range(10)}

        paired_eeg, paired_images = pipeline.create_paired_dataset(
            test_input, labels, digit_images
        )
        assert len(paired_eeg) == 5
        assert len(paired_images) == 5
        print("   ✅ Image reconstruction pipeline works correctly")

        print("✅ Comprehensive validation passed!")
        return True

    except Exception as e:
        print(f"❌ Comprehensive validation failed: {e}")
        return False


def main():
    """Main test runner"""
    print("🧠 EEG RECONSTRUCTION PROJECT - TEST RUNNER")
    print("=" * 50)

    start_time = time.time()

    # Test sequence
    tests = [
        ("Dependency Check", check_dependencies),
        ("Import Test", run_import_test),
        ("Quick Demo", run_quick_demo),
        ("Comprehensive Validation", run_comprehensive_validation),
        ("Unit Tests", run_unit_tests),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        results[test_name] = test_func()

        if not results[test_name]:
            print(f"\n❌ {test_name} failed! Stopping here.")
            break

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} {status}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.2f} seconds")

    # Overall result
    all_passed = all(results.values())

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The EEG reconstruction project is ready to use!")

        # Optionally run comprehensive demo
        user_input = input("\nRun comprehensive demo? (y/N): ").strip().lower()
        if user_input == 'y':
            run_comprehensive_demo()
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the errors above and fix them.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
