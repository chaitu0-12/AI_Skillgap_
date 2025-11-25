import unittest
import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

class TestSkillGapAnalyzer(unittest.TestCase):
    
    def test_extract_text_from_txt(self):
        """Test extraction of text from TXT file"""
        from app.main import extract_text_from_txt
        import io
        
        # Create a simple text file content
        text_content = "This is a test resume with Python and JavaScript skills."
        text_file = io.BytesIO(text_content.encode('utf-8'))
        
        # Extract text
        extracted_text = extract_text_from_txt(text_file)
        
        # Verify the text was extracted correctly
        self.assertEqual(extracted_text.strip(), text_content)
    
    def test_compute_similarity(self):
        """Test computation of similarity between skill sets"""
        from app.main import compute_similarity
        
        # Test with identical skill sets
        resume_skills = ["Python", "JavaScript", "SQL"]
        jd_skills = ["Python", "JavaScript", "SQL"]
        
        similarity = compute_similarity(resume_skills, jd_skills)
        
        # Identical skill sets should have 100% similarity
        self.assertAlmostEqual(similarity, 100.0, places=10)
        
        # Test with completely different skill sets
        resume_skills = ["Python", "JavaScript"]
        jd_skills = ["Project Management", "Marketing"]
        
        similarity = compute_similarity(resume_skills, jd_skills)
        
        # Different skill sets should have 0% similarity
        self.assertAlmostEqual(similarity, 0.0, places=10)
    
    def test_categorize_skills(self):
        """Test categorization of skills"""
        from app.main import categorize_skills
        
        resume_skills = ["Python", "JavaScript", "SQL"]
        jd_skills = ["Python", "Java", "SQL", "React"]
        
        matched, partial, missing = categorize_skills(resume_skills, jd_skills)
        
        # Check matched skills
        self.assertIn("Python", matched)
        self.assertIn("SQL", matched)
        
        # Check missing skills
        self.assertIn("Java", missing)
        self.assertIn("React", missing)

if __name__ == '__main__':
    unittest.main()