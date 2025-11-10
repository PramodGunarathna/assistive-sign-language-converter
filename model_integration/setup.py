"""
Setup script for Sign Language Recognition System - New Project
"""
import os
import sys
import subprocess

def create_directories():
    """Create necessary directories"""
    directories = [
        'src/models/trained',
        'src/utils',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_requirements():
    """Install Python requirements"""
    try:
        print("Installing Python requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing requirements: {e}")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = [
        'tensorflow',
        'opencv-python',
        'mediapipe',
        'numpy',
        'google-generativeai'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def main():
    """Main setup function"""
    print("Setting up Sign Language Recognition System - New Project")
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install requirements
    print("\n2. Installing requirements...")
    if not install_requirements():
        print("Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return
    
    # Check dependencies
    print("\n3. Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Please install them manually:")
        for package in missing:
            print(f"pip install {package}")
    else:
        print("\n✅ All dependencies are available!")
    
    print("\n" + "=" * 60)
    print("Setup completed!")
    print("\nTo run the application:")
    print("python main.py")
    print("\nMake sure to:")
    print("1. Place your trained models in 'src/models/trained/'")
    print("2. Update the Gemini API key in 'src/sentence_prediction.py'")
    print("3. Ensure your camera is working")

if __name__ == "__main__":
    main()