# AI Role-Based Skill Gap Analyzer with Flask + React - Implementation Summary

## Project Overview

The AI Role-Based Skill Gap Analyzer is a full-stack web application that analyzes skill gaps between your resume and predefined role requirements using NLP and machine learning techniques. Instead of uploading job descriptions, users select their target role from a list of predefined roles. The application helps job seekers identify missing skills and provides recommendations for upskilling.

## Technology Stack

### Backend
- **Language**: Python
- **Web Framework**: Flask
- **NLP & ML Models**: 
  - spaCy (for text processing and Named Entity Recognition)
  - Sentence-BERT (Transformers from Hugging Face)
  - Scikit-learn (for cosine similarity calculations)
- **Document Parsing**:
  - pdfplumber (for PDFs)
  - python-docx (for DOCX)
  - Built-in methods (for TXT)
- **Authentication**: python-dotenv for Hugging Face token management

### Frontend
- **Framework**: React
- **State Management**: React Hooks
- **HTTP Client**: Axios
- **Visualization Libraries**: Chart.js
- **Styling**: Tailwind CSS

## Key Features Implemented

1. **Document Parsing**
   - Upload and parse PDF, DOCX, and TXT files for resumes
   - Extract text content from various document formats

2. **Role-Based Skill Requirements**
   - Predefined skill requirements for 8 professional roles
   - Technical and soft skills categorized for each role

3. **Skill Extraction**
   - Extract technical and soft skills using spaCy NLP processing
   - Custom keyword-based extraction for domain-specific skills
   - Categorize skills into technical and soft skills

4. **Skill Gap Analysis**
   - Compute cosine similarity between resume and role requirements
   - Identify matched, partially matched, and missing skills
   - Calculate overall match percentage

5. **Visualization**
   - Interactive dashboard with improved theme styling
   - Bar chart for skill match overview
   - Donut chart for overall match percentage
   - Tag-based display of skills with color coding

6. **Recommendations**
   - Generate upskilling recommendations based on missing skills
   - Provide actionable suggestions for skill development
   - Role-specific learning paths

7. **Export Functionality**
   - Export results as CSV for further analysis
   - Role-specific filenames for exported files

## File Structure

```
infosys_resume_analyzer/
├── app/
│   ├── backend.py           # Flask backend application
│   ├── frontend/            # React frontend application
│   │   ├── public/          # Public assets
│   │   ├── src/             # Source code
│   │   ├── package.json     # Frontend dependencies
│   │   └── README.md        # Frontend documentation
│   ├── roles_data.py        # Role skills data
│   └── huggingface_auth.py  # Hugging Face authentication
├── sample_resume.txt        # Sample resume for testing
├── sample_job_description.txt # Sample job description for testing
├── requirements.txt         # Python dependencies
├── run.py                   # Main run script
├── build_frontend.py        # Frontend build script
├── test_app.py              # Unit tests
├── test_enhanced.py         # Enhanced tests
├── IMPLEMENTATION_SUMMARY.md # Implementation details
├── .env                     # Environment variables
└── README.md               # This file
```

## How to Run the Application

1. **Prerequisites**:
   - Python 3.7 or higher
   - pip package manager
   - Node.js and npm

2. **Installation**:
   ```bash
   # Install required Python packages
   pip install -r requirements.txt
   
   # Download spaCy English model
   python -m spacy download en_core_web_sm
   
   # Install Node.js dependencies
   cd app/frontend
   npm install
   ```

3. **Running the Application**:
   ```bash
   # Development mode
   python run.py dev
   
   # Production mode
   python run.py
   ```

## Testing

The application includes unit tests for core functionality:
- Text extraction from TXT files
- Skill similarity computation
- Skill categorization

Run tests with:
```bash
python test_app.py
```

## Customization

The application can be easily customized by:
- Modifying the technical and soft skills keywords in the `extract_skills` function
- Adjusting the recommendation mapping in the `get_recommendations` function
- Updating the CSS styling in the `st.markdown` section

## Deployment Options

The application can be deployed to:
- Heroku
- AWS
- Google Cloud Platform
- Microsoft Azure
- Other cloud platforms that support Python and Node.js applications

## Future Enhancements

1. **Advanced NLP Processing**:
   - Implement more sophisticated NER models
   - Add support for industry-specific terminology

2. **Enhanced Matching Algorithm**:
   - Use semantic similarity with Sentence-BERT embeddings
   - Implement fuzzy matching for similar skills

3. **Additional Features**:
   - PDF export functionality
   - User authentication and profiles
   - Historical analysis and progress tracking
   - Add more professional roles
   - Custom role creation

4. **Improved UI/UX**:
   - Add more interactive visualizations
   - Implement responsive design for mobile devices
   - Add dark mode support
   - Role comparison feature

5. **Hugging Face Integration**:
   - Enhanced model loading with authentication
   - Model fine-tuning capabilities
   - Integration with Hugging Face Hub for sharing results

This implementation provides a solid foundation for a skill gap analysis tool that can be extended with additional features and improvements based on user feedback and requirements.