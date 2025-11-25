import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed required packages.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def download_spacy_model():
    """Download the spaCy English model"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully downloaded spaCy English model.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading spaCy model: {e}")
        sys.exit(1)

def run_app():
    """Run the Streamlit application"""
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app/main.py"])
    except subprocess.CalledProcessError as e:
        print(f"Error running the application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("AI-Powered Skill Gap Analyzer Setup")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found.")
        sys.exit(1)
    
    # Install requirements
    print("Installing required packages...")
    install_requirements()
    
    # Download spaCy model
    print("Downloading spaCy English model...")
    download_spacy_model()
    
    print("\nSetup complete!")
    print("To run the application, execute: python setup.py run")
    
    # Check if user wants to run the app
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("Starting the application...")
        run_app()