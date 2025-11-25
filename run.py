import os
import subprocess
import sys
import threading
import time

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("Python dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("Error: Failed to install Python dependencies")
        return False
    return True

def start_backend():
    """Start the Flask backend"""
    print("Starting Flask backend...")
    backend_dir = os.path.join(os.path.dirname(__file__), 'app')
    try:
        subprocess.run([sys.executable, "-m", "app.flask_backend"], cwd=os.path.dirname(__file__), check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to start Flask backend")
        return False
    return True

def start_frontend():
    """Start the React frontend"""
    print("Starting React frontend...")
    frontend_dir = os.path.join(os.path.dirname(__file__), 'app', 'frontend')
    try:
        subprocess.run(['npm', 'start'], cwd=frontend_dir, check=True)
    except subprocess.CalledProcessError:
        print("Error: Failed to start React frontend")
        return False
    return True

def build_and_serve_frontend():
    """Build and serve the React frontend"""
    print("Building React frontend...")
    frontend_dir = os.path.join(os.path.dirname(__file__), 'app', 'frontend')
    try:
        # Build the frontend
        subprocess.run(['npm', 'run', 'build'], cwd=frontend_dir, check=True)
        print("Frontend built successfully!")
    except subprocess.CalledProcessError:
        print("Error: Failed to build React frontend")
        return False
    return True

if __name__ == "__main__":
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check if we're in development mode
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=start_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Give backend time to start
        time.sleep(3)
        
        # Start frontend
        start_frontend()
    else:
        # Build and serve frontend
        if build_and_serve_frontend():
            # Start backend
            start_backend()