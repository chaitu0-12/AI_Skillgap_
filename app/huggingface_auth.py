import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_huggingface_token():
    """Get Hugging Face token from environment variables"""
    return os.getenv("HUGGINGFACE_TOKEN")

def is_huggingface_authenticated():
    """Check if Hugging Face token is available"""
    token = get_huggingface_token()
    return token is not None and token != "your_huggingface_token_here"

def authenticate_huggingface():
    """Authenticate with Hugging Face using token"""
    token = get_huggingface_token()
    if not token or token == "your_huggingface_token_here":
        raise ValueError("Hugging Face token not found in environment variables")
    
    # In a real implementation, you would use the token to authenticate
    # For example: HfFolder.save_token(token)
    return token