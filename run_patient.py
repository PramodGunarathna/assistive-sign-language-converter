#!/usr/bin/env python3
"""
Patient Client Launcher
Simple launcher script for the patient communication client.
"""

import sys
import os

# Add the client directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'client'))

try:
    from patient_client import PatientClient
    
    if __name__ == "__main__":
        print("Starting Patient Communication Client...")
        print("Make sure you have the doctor's IP address.")
        print("-" * 50)
        
        app = PatientClient()
        app.run()
        
except ImportError as e:
    print(f"Error importing patient client: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error starting patient client: {e}")
    sys.exit(1)
