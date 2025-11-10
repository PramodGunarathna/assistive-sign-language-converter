#!/usr/bin/env python3
"""
Doctor Server Launcher
Simple launcher script for the doctor communication server.
"""

import sys
import os

# Add the doctor directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'doctor'))

try:
    from server import DoctorServer
    
    if __name__ == "__main__":
        print("Starting Doctor Communication Server...")
        print("Make sure your firewall allows Python to accept connections.")
        print("-" * 50)
        
        app = DoctorServer()
        app.run()
        
except ImportError as e:
    print(f"Error importing doctor server: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)
except Exception as e:
    print(f"Error starting doctor server: {e}")
    sys.exit(1)
