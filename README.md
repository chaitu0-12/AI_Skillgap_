# AI Role-Based Skill Gap Analyzer

This application analyzes skill gaps between your resume and predefined role requirements using NLP and machine learning techniques. The application uses a Flask backend with a React frontend instead of Streamlit.

## Features

- Upload and parse PDF, DOCX, and TXT files
- Select from 8 predefined professional roles (Data Analyst, Software Engineer, ML Engineer, etc.)
- Extract technical and soft skills using spaCy and custom NER
- Compare skills using Sentence-BERT embeddings
- Calculate cosine similarity between your resume and role requirements
- Identify and rank missing skills
- Visualize results via Streamlit dashboard with improved theme
- Export results as CSV
- Hugging Face authentication support

## Technology Stack

### Backend
- **Language**: Python
- **Web Framework**: Flask
- **NLP & ML Models**: 
  - spaCy (for text processing and NER)
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

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd infosys_resume_analyzer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_sm
   ```

5. Install Node.js dependencies for the frontend:
   ```
   cd app/frontend
   npm install
   ```

## Usage

### Development Mode
To run both the backend and frontend in development mode:
```
python run.py dev
```

This will start:
- Flask backend on http://localhost:5000
- React frontend on http://localhost:3000

### Production Mode
To build and serve the application in production mode:
```
python run.py
```

This will:
- Build the React frontend
- Start the Flask backend which serves the built frontend

## How to Use

1. Upload your resume (PDF, DOCX, or TXT format)
2. Select your target role from the dropdown menu
3. Click "Analyze Skills"
4. Review the extracted skills, skill gap analysis, and recommendations
5. Export results as CSV if needed

### Hugging Face Setup

1. Get your Hugging Face token from https://huggingface.co/settings/tokens
2. Add it to the `.env` file:
   ```
   HUGGINGFACE_TOKEN=your_actual_token_here
   ```

## Project Structure

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

## Customization

You can customize the role requirements by modifying the `ROLE_SKILLS` dictionary in `app/roles_data.py`.

You can also customize the skill extraction by modifying the technical and soft skills keywords in the `extract_skills` function in `app/main.py`.

## Deployment

This application can be deployed to:
- Streamlit Cloud
- Heroku
- AWS
- Other cloud platforms that support Python applications

## License

This project is licensed under the MIT License.