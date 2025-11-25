import os
import subprocess
import sys

def build_frontend():
    """Build the React frontend"""
    frontend_dir = os.path.join(os.path.dirname(__file__), 'app', 'frontend')
    
    # Check if Node.js is installed
    try:
        subprocess.run(['node', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Node.js is not installed or not in PATH")
        return False
    
    # Check if npm is installed
    try:
        subprocess.run(['npm', '--version'], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: npm is not installed or not in PATH")
        return False
    
    # Install dependencies
    print("Installing frontend dependencies...")
    try:
        subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to install frontend dependencies")
        return False
    
    # Build the frontend
    print("Building frontend...")
    try:
        subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to build frontend")
        return False
    
    print("Frontend built successfully!")
    return True

if __name__ == "__main__":
    build_frontend()