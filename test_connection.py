#!/usr/bin/env python3
"""
Connection Test Script
Simple test to verify network connectivity and basic socket functionality.
"""

import socket
import sys
import threading
import time

def test_socket_server():
    """Test basic socket server functionality."""
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind(('localhost', 12346))  # Use different port for testing
        server_socket.listen(1)
        
        print("✓ Socket server test passed")
        server_socket.close()
        return True
    except Exception as e:
        print(f"✗ Socket server test failed: {e}")
        return False

def test_socket_client():
    """Test basic socket client functionality."""
    try:
        # Start a simple server in a thread
        def simple_server():
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('localhost', 12347))
            server_socket.listen(1)
            
            conn, addr = server_socket.accept()
            conn.close()
            server_socket.close()
        
        server_thread = threading.Thread(target=simple_server, daemon=True)
        server_thread.start()
        
        time.sleep(0.1)  # Give server time to start
        
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', 12347))
        client_socket.close()
        
        print("✓ Socket client test passed")
        return True
    except Exception as e:
        print(f"✗ Socket client test failed: {e}")
        return False

def test_tkinter():
    """Test tkinter availability."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # Hide the window
        root.destroy()
        print("✓ Tkinter test passed")
        return True
    except Exception as e:
        print(f"✗ Tkinter test failed: {e}")
        return False

def test_json():
    """Test JSON functionality."""
    try:
        import json
        test_data = {"test": "data", "number": 123}
        json_str = json.dumps(test_data)
        parsed_data = json.loads(json_str)
        assert parsed_data == test_data
        print("✓ JSON test passed")
        return True
    except Exception as e:
        print(f"✗ JSON test failed: {e}")
        return False

def get_local_ip():
    """Get and display local IP address."""
    try:
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_socket.connect(("8.8.8.8", 80))
        local_ip = temp_socket.getsockname()[0]
        temp_socket.close()
        print(f"✓ Local IP address: {local_ip}")
        return local_ip
    except Exception as e:
        print(f"✗ Could not determine local IP: {e}")
        return "127.0.0.1"

def main():
    """Run all tests."""
    print("Doctor-Patient Communication System - Connection Test")
    print("=" * 55)
    print()
    
    tests_passed = 0
    total_tests = 5
    
    print("Running basic functionality tests...")
    print("-" * 35)
    
    if test_socket_server():
        tests_passed += 1
    
    if test_socket_client():
        tests_passed += 1
    
    if test_tkinter():
        tests_passed += 1
    
    if test_json():
        tests_passed += 1
    
    get_local_ip()
    tests_passed += 1  # IP detection is informational
    
    print()
    print("Test Results:")
    print("-" * 15)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! The system should work correctly.")
        print()
        print("Next steps:")
        print("1. Run 'python run_doctor.py' on the doctor's computer")
        print("2. Note the IP address displayed by the doctor application")
        print("3. Run 'python run_patient.py' on the patient's computer")
        print("4. Enter the doctor's IP address to connect")
    else:
        print("✗ Some tests failed. Please check your Python installation.")
        print("Required modules: socket, threading, tkinter, json, datetime")
        sys.exit(1)

if __name__ == "__main__":
    main()
