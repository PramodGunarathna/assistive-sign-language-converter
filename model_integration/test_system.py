#!/usr/bin/env python3
"""
Test script for Sign Language Recognition System - New Project
"""
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.real_time_prediction import RealTimePredictor
        print("✓ Real-time prediction module imported")
    except Exception as e:
        print(f"✗ Real-time prediction import failed: {e}")
        return False
    
    try:
        from src.sentence_prediction import SentencePredictor
        print("✓ Sentence prediction module imported")
    except Exception as e:
        print(f"✗ Sentence prediction import failed: {e}")
        return False
    
    try:
        from src.utils.model_utils import SignLanguageModel
        print("✓ Model utilities imported")
    except Exception as e:
        print(f"✗ Model utilities import failed: {e}")
        return False
    
    try:
        from src.utils.robust_mediapipe import RobustMediaPipeProcessor
        print("✓ MediaPipe processor imported")
    except Exception as e:
        print(f"✗ MediaPipe processor import failed: {e}")
        return False
    
    try:
        from gemini_integration import SignLanguageLLMProcessor
        print("✓ Gemini integration imported")
    except Exception as e:
        print(f"✗ Gemini integration import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation with different architectures"""
    print("\nTesting model creation...")
    
    try:
        from src.utils.model_utils import SignLanguageModel
        
        # Test basic architecture
        model_basic = SignLanguageModel(['hello', 'world'], model_type='basic')
        print("✓ Basic model created")
        
        # Test improved architecture
        model_improved = SignLanguageModel(['hello', 'world'], model_type='improved')
        print("✓ Improved model created")
        
        # Test advanced architecture
        model_advanced = SignLanguageModel(['hello', 'world'], model_type='advanced')
        print("✓ Advanced model created")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_gemini_integration():
    """Test Gemini integration"""
    print("\nTesting Gemini integration...")
    
    try:
        from gemini_integration import SignLanguageLLMProcessor
        
        # Test with a dummy API key (will fail but should not crash)
        processor = SignLanguageLLMProcessor("dummy_key")
        print("✓ Gemini processor created (with dummy key)")
        
        # Test basic enhancement
        result = processor._basic_enhance_sentence("pain")
        print(f"✓ Basic enhancement works: '{result}'")
        
        return True
    except Exception as e:
        print(f"✗ Gemini integration test failed: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe processor"""
    print("\nTesting MediaPipe processor...")
    
    try:
        from src.utils.robust_mediapipe import RobustMediaPipeProcessor
        
        processor = RobustMediaPipeProcessor()
        print("✓ MediaPipe processor created")
        
        # Test cleanup
        processor.close()
        print("✓ MediaPipe processor closed")
        
        return True
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Sign Language Recognition System - Test Suite")
    print("=" * 50)
    
    all_tests_passed = True
    
    # Test imports
    if not test_imports():
        all_tests_passed = False
    
    # Test model creation
    if not test_model_creation():
        all_tests_passed = False
    
    # Test Gemini integration
    if not test_gemini_integration():
        all_tests_passed = False
    
    # Test MediaPipe
    if not test_mediapipe():
        all_tests_passed = False
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("✅ All tests passed! System is ready to use.")
        print("\nTo run the application:")
        print("python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
