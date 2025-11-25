import unittest
import sys
import os
import json

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

class TestFlaskBackend(unittest.TestCase):
    
    def test_backend_imports(self):
        """Test that Flask backend can be imported without errors"""
        try:
            from app.flask_backend import app, extract_skills, get_role_skills
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import Flask backend: {e}")
    
    def test_role_skills_data(self):
        """Test that role skills data is properly loaded"""
        from app.roles_data import ROLE_SKILLS
        
        # Check that we have the expected roles
        expected_roles = [
            "Data Analyst", "Software Engineer", "ML Engineer", 
            "Full Stack Developer", "DevOps Engineer", "Product Manager",
            "UI/UX Designer", "Cybersecurity Analyst"
        ]
        
        for role in expected_roles:
            self.assertIn(role, ROLE_SKILLS)
            self.assertIn("technical_skills", ROLE_SKILLS[role])
            self.assertIn("soft_skills", ROLE_SKILLS[role])
    
    def test_get_role_skills(self):
        """Test getting skills for a specific role"""
        from app.flask_backend import get_role_skills
        
        # Test with a valid role
        tech_skills, soft_skills = get_role_skills("Data Analyst")
        self.assertGreater(len(tech_skills), 0)
        self.assertGreater(len(soft_skills), 0)
        
        # Test with an invalid role
        tech_skills, soft_skills = get_role_skills("Invalid Role")
        self.assertEqual(len(tech_skills), 0)
        self.assertEqual(len(soft_skills), 0)
    
    def test_huggingface_auth(self):
        """Test Hugging Face authentication functions"""
        from app.huggingface_auth import get_huggingface_token, is_huggingface_authenticated
        
        # Test that we can get the token
        token = get_huggingface_token()
        self.assertIsNotNone(token)
        
        # Test authentication check (will be False with default token)
        # In a real test, we would set a valid token
        is_authenticated = is_huggingface_authenticated()
        # This will be False with the default token
        self.assertFalse(is_authenticated)

if __name__ == '__main__':
    unittest.main()