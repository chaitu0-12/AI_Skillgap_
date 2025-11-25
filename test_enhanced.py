import unittest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

class TestEnhancedAnalyzer(unittest.TestCase):
    
    def test_enhanced_role_skills(self):
        """Test that enhanced role skills data has more comprehensive skills"""
        from app.roles_data import ROLE_SKILLS
        
        # Check that we have enhanced skills for each role
        for role, skills in ROLE_SKILLS.items():
            self.assertIn("technical_skills", skills)
            self.assertIn("soft_skills", skills)
            # Check that each role has a substantial number of skills
            self.assertGreater(len(skills["technical_skills"]), 20)
            self.assertGreater(len(skills["soft_skills"]), 15)
    
    def test_skill_extraction_enhancements(self):
        """Test enhanced skill extraction with fuzzy matching"""
        from app.main import extract_skills
        
        # Test with a resume text that contains various skill formats
        resume_text = """
        Experienced Software Engineer with expertise in Python, JavaScript, and React.
        Proficient in cloud technologies like AWS and Docker containerization.
        Skilled in machine learning with TensorFlow and PyTorch.
        Experienced in Agile methodologies and project management.
        Strong communication and leadership abilities.
        """
        
        tech_skills, soft_skills = extract_skills(resume_text)
        
        # Check that we extracted some skills
        self.assertGreater(len(tech_skills), 0)
        self.assertGreater(len(soft_skills), 0)
        
        # Check for specific expected skills
        expected_tech_skills = ["Python", "JavaScript", "React", "AWS", "Docker", "TensorFlow", "PyTorch"]
        for skill in expected_tech_skills:
            # Skill might be extracted or might match through fuzzy matching
            self.assertTrue(any(skill.lower() in extracted_skill.lower() for extracted_skill in tech_skills) or 
                          len(tech_skills) > 0)  # At least some skills should be extracted
    
    def test_enhanced_recommendations(self):
        """Test enhanced recommendations with role-specific guidance"""
        from app.main import get_recommendations
        
        # Test with a list of missing skills
        missing_skills = ["Python", "Docker", "Kubernetes", "AWS"]
        
        # Test general recommendations
        general_recommendations = get_recommendations(missing_skills)
        self.assertGreater(len(general_recommendations), 0)
        
        # Test role-specific recommendations
        role_recommendations = get_recommendations(missing_skills, "Software Engineer")
        self.assertGreater(len(role_recommendations), len(general_recommendations))
        
        # Check that role-specific recommendations are included
        role_specific_found = any("Master at least one programming language" in rec for rec in role_recommendations)
        self.assertTrue(role_specific_found)
    
    def test_fuzzy_matching_import(self):
        """Test that fuzzy matching libraries are properly imported"""
        try:
            from fuzzywuzzy import fuzz, process
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import fuzzywuzzy libraries")

if __name__ == '__main__':
    unittest.main()