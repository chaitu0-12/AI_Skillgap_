import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import docx
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import re
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process

# Load environment variables
load_dotenv()

# Hugging Face authentication
from app.huggingface_auth import is_huggingface_authenticated, authenticate_huggingface

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("Please install the spaCy English model: python -m spacy download en_core_web_sm")
    st.stop()

# Load role-based skills data
from app.roles_data import ROLE_SKILLS

# Hugging Face authentication check
if not is_huggingface_authenticated():
    st.warning("Hugging Face token not found. Please set your HUGGINGFACE_TOKEN in the .env file for full functionality.")
else:
    try:
        token = authenticate_huggingface()
        st.success("Successfully authenticated with Hugging Face")
    except Exception as e:
        st.error(f"Hugging Face authentication failed: {str(e)}")

# Load Sentence-BERT model
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_sentence_model()

# Professional theme with enhanced UI/UX
st.markdown(
    """
    <style>
    /* Main layout */
    .main {
        background-color: #FFFFFF;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.8rem;
        color: #111827;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .stCard, .stExpander {
        background-color: #F9FAFB;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .stCard:hover, .stExpander:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #C7D2FE;
    }
    
    /* Skill tags */
    .skill-tag {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 0.4rem 0.8rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        border: 1px solid #C7D2FE;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .skill-tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }
    
    .soft-skill {
        background-color: #F0FDF4;
        color: #16A34A;
        border: 1px solid #BBF7D0;
    }
    
    .missing-skill {
        background-color: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
    }
    
    .partial-skill {
        background-color: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
    }
    
    /* Section headers */
    .section-header {
        color: #111827;
        margin-top: 2rem;
        margin-bottom: 1.2rem;
        font-weight: 700;
        font-size: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #F9FAFB;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
    }
    
    .stMetric:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.08);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #F9FAFB;
        border: 1px dashed #E5E7EB;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Plotly charts */
    .stPlotlyChart {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    /* Role cards */
    .role-preview {
        background-color: #F9FAFB;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .role-preview:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-color: #C7D2FE;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F9FAFB;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #D1D5DB;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #9CA3AF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Main header
st.markdown('<p class="main-header">AI Role-Based Skill Gap Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your resume and select your target role to identify skill gaps and get upskilling recommendations.</p>', unsafe_allow_html=True)

# File parsing functions
def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file):
    """Extract text from TXT file"""
    return file.read().decode("utf-8")

def get_role_skills(role):
    """Get skills for a specific role"""
    if role in ROLE_SKILLS:
        return ROLE_SKILLS[role]["technical_skills"], ROLE_SKILLS[role]["soft_skills"]
    return [], []

def extract_skills(text):
    """Extract skills from text using spaCy and custom rules with improved accuracy"""
    # Process text with spaCy
    doc = nlp(text.lower())
    
    # Enhanced technical skills keywords with variations and synonyms
    technical_skills_keywords = [
        # Programming Languages
        "python", "java", "javascript", "c++", "c#", "sql", "html", "css", "php", "ruby",
        "go", "golang", "rust", "scala", "kotlin", "swift", "r", "matlab", "typescript",
        "shell", "bash", "powershell", "perl", "lua", "dart", "elixir", "erlang", "haskell",
        "clojure", "groovy", "julia", "fortran", "cobol", "assembly", "vba", "vb.net",
        
        # Web Technologies
        "react", "angular", "vue", "vue.js", "node.js", "express", "django", "flask", 
        "spring", "spring boot", "hibernate", "asp.net", ".net", "laravel", "rails", "ruby on rails",
        "next.js", "nuxt.js", "svelte", "ember.js", "backbone.js", "jquery", "bootstrap", 
        "tailwind", "tailwind css", "sass", "scss", "less", "webpack", "vite", "gulp", "grunt",
        "npm", "yarn", "pnpm", "rest", "restful", "rest api", "graphql", "soap", "microservices",
        "api", "json", "xml", "ajax", "axios", "fetch", "cors", "seo", "pwa", "ssr", "csr",
        
        # Databases
        "mongodb", "postgresql", "postgres", "mysql", "sqlite", "oracle", "sql server",
        "mssql", "redis", "elasticsearch", "cassandra", "couchbase", "dynamodb", "firebase",
        "neo4j", "influxdb", "cockroachdb", "snowflake", "redshift", "bigquery", "athena",
        "databricks", "data warehouse", "data lake", "etl", "elt", "olap", "oltp",
        
        # Cloud & DevOps
        "docker", "kubernetes", "k8s", "aws", "amazon web services", "azure", "gcp", "google cloud",
        "cloud", "serverless", "lambda", "ec2", "s3", "rds", "eks", "ecs", "fargate",
        "cloudformation", "terraform", "ansible", "chef", "puppet", "jenkins", "ci/cd",
        "github actions", "gitlab ci", "circleci", "travisci", "bamboo", "teamcity",
        "devops", "sre", "site reliability", "monitoring", "logging", "observability",
        "prometheus", "grafana", "datadog", "new relic", "splunk", "elk", "elasticsearch",
        "logstash", "kibana", "nagios", "zabbix", "sentry", "pagerduty", "opsgenie",
        
        # Machine Learning & AI
        "tensorflow", "pytorch", "scikit-learn", "keras", "pandas", "numpy", "matplotlib",
        "seaborn", "plotly", "bokeh", "jupyter", "jupyter notebook", "machine learning",
        "ml", "deep learning", "neural networks", "nlp", "natural language processing",
        "computer vision", "cv", "reinforcement learning", "rl", "data science", "data analysis",
        "data engineering", "big data", "hadoop", "spark", "kafka", "flink", "storm",
        "airflow", "mlflow", "kubeflow", "fastapi", "streamlit", "gradio", "hugging face",
        "transformers", "bert", "gpt", "llm", "large language model", "chatgpt", "openai",
        
        # Data & Analytics
        "excel", "tableau", "power bi", "looker", "qlik", "qliksense", "sisense", "mode",
        "metabase", "redash", "superset", "analytics", "business intelligence", "bi",
        "statistics", "statistical", "r", "sas", "spss", "matlab", "stata", "data mining",
        "data visualization", "dashboard", "kpi", "metrics", "reporting", "ab testing",
        "a/b testing", "hypothesis testing", "regression", "clustering", "classification",
        
        # Tools & Platforms
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "slack", "teams",
        "trello", "asana", "monday.com", "notion", "figma", "sketch", "adobe xd", "zeplin",
        "postman", "insomnia", "soapui", "wireshark", "burp suite", "owasp", "sonarqube",
        "linux", "unix", "windows", "macos", "ubuntu", "centos", "redhat", "debian",
        "fedora", "suse", "alpine", "bash", "shell scripting", "powershell scripting",
        "agile", "scrum", "kanban", "waterfall", "lean", "six sigma", "project management",
        "pmp", "prince2", "safe", "scaled agile", "product management", "product owner",
        "product manager", "ux", "ui", "user experience", "user interface", "design thinking",
        
        # Security
        "cybersecurity", "security", "penetration testing", "pentesting", "vulnerability",
        "vulnerability assessment", "siem", "ids", "ips", "firewall", "encryption",
        "ssl", "tls", "pkI", "authentication", "authorization", "oauth", "saml", "jwt",
        "zero trust", "compliance", "gdpr", "hipaa", "pci dss", "iso 27001", "nist",
        "soc", "incident response", "forensics", "malware analysis", "threat hunting",
        "threat intelligence", "risk management", "risk assessment", "auditing"
    ]
    
    # Enhanced soft skills keywords with variations
    soft_skills_keywords = [
        "communication", "leadership", "teamwork", "collaboration", "problem solving",
        "critical thinking", "analytical thinking", "creativity", "innovation", "adaptability",
        "flexibility", "time management", "organization", "interpersonal skills", "people skills",
        "emotional intelligence", "eq", "decision making", "conflict resolution", "negotiation",
        "persuasion", "influence", "project management", "mentoring", "coaching", "training",
        "public speaking", "presentation", "writing", "technical writing", "documentation",
        "research", "analytical", "detail oriented", "attention to detail", "customer service",
        "client service", "sales", "marketing", "strategic planning", "strategic thinking",
        "budgeting", "financial management", "risk management", "quality assurance", "qa",
        "facilitation", "change management", "process improvement", "continuous improvement",
        "kaizen", "lean", "six sigma", "entrepreneurship", "business acumen", "market awareness",
        "competitive intelligence", "stakeholder management", "vendor management", "relationship building",
        "networking", "team building", "motivation", "inspiration", "empowerment", "delegation",
        "accountability", "responsibility", "ownership", "resilience", "persistence", "tenacity",
        "patience", "tolerance", "open-mindedness", "curiosity", "learning agility", "adaptability",
        "agility", "versatility", "multitasking", "prioritization", "work ethic", "integrity",
        "ethics", "honesty", "reliability", "dependability", "trustworthiness", "loyalty",
        "commitment", "dedication", "passion", "enthusiasm", "optimism", "positivity",
        "confidence", "self-confidence", "assertiveness", "diplomacy", "tact", "discretion",
        "discretionary", "confidentiality", "secrecy", "privacy", "security awareness", "compliance",
        "regulatory", "legal", "governance", "corporate governance", "corporate social responsibility",
        "csr", "sustainability", "environmental", "social", "governance", "esg", "diversity",
        "inclusion", "equity", "accessibility", "multicultural", "global", "international",
        "cross-cultural", "cultural awareness", "cultural sensitivity", "multilingual", "language skills"
    ]
    
    # Extract named entities
    entities = [ent.text.lower() for ent in doc.ents]
    
    # Extract noun chunks
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    
    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    # Combine all text elements
    all_text_elements = entities + noun_chunks + text.lower().split() + lemmatized_tokens
    
    # Find technical skills with fuzzy matching
    technical_skills = []
    for skill in technical_skills_keywords:
        # Exact match
        if any(skill in element for element in all_text_elements):
            technical_skills.append(skill.title())
        # Partial match for compound skills
        elif any(len(skill.split()) > 1 and all(word in element for word in skill.split()) for element in all_text_elements):
            technical_skills.append(skill.title())
        # Fuzzy match for similar skills (threshold 80)
        else:
            matches = process.extract(skill, all_text_elements, limit=1)
            if matches and matches[0][1] >= 80:
                technical_skills.append(skill.title())
    
    # Find soft skills with fuzzy matching
    soft_skills = []
    for skill in soft_skills_keywords:
        # Exact match
        if any(skill in element for element in all_text_elements):
            soft_skills.append(skill.title())
        # Partial match for compound skills
        elif any(len(skill.split()) > 1 and all(word in element for word in skill.split()) for element in all_text_elements):
            soft_skills.append(skill.title())
        # Fuzzy match for similar skills (threshold 80)
        else:
            matches = process.extract(skill, all_text_elements, limit=1)
            if matches and matches[0][1] >= 80:
                soft_skills.append(skill.title())
    
    # Remove duplicates and sort
    technical_skills = sorted(list(set(technical_skills)))
    soft_skills = sorted(list(set(soft_skills)))
    
    return technical_skills, soft_skills

def compute_similarity(resume_skills, jd_skills):
    """Compute similarity between resume skills and job description skills"""
    # Combine all skills
    all_skills = list(set(resume_skills + jd_skills))
    
    # Create binary vectors (1 if skill present, 0 otherwise)
    resume_vector = [1 if skill in resume_skills else 0 for skill in all_skills]
    jd_vector = [1 if skill in jd_skills else 0 for skill in all_skills]
    
    # Compute cosine similarity
    similarity = cosine_similarity([resume_vector], [jd_vector])[0][0]
    
    return similarity * 100  # Convert to percentage

def categorize_skills(resume_skills, jd_skills):
    """Categorize skills as matched, partial, or missing"""
    matched_skills = list(set(resume_skills) & set(jd_skills))
    missing_skills = list(set(jd_skills) - set(resume_skills))
    
    # For this implementation, we'll consider partial matches as skills that are similar
    # but not exactly matched. In a more advanced version, we could use semantic similarity.
    partial_skills = []
    
    return matched_skills, partial_skills, missing_skills

def get_recommendations(missing_skills, role=None):
    """Generate upskilling recommendations based on missing skills and role"""
    recommendations = []
    
    # Comprehensive mapping of skills to recommendations
    recommendation_map = {
        # Programming Languages
        "Python": "Take Python courses on Coursera, edX, or Udemy. Practice on LeetCode, HackerRank, or Codewars. Build projects on GitHub to showcase your skills.",
        "Java": "Learn Java through Oracle's official tutorials and Oracle's Java certification paths. Build enterprise applications and Android apps to practice.",
        "JavaScript": "Practice JavaScript on freeCodeCamp, Codecademy, or Frontend Mentor. Build interactive web projects and learn modern frameworks like React or Vue.js.",
        "C++": "Study C++ through books like 'The C++ Programming Language' or online courses. Practice on competitive programming platforms.",
        "C#": "Learn C# through Microsoft's official documentation and build .NET applications. Consider Unity for game development practice.",
        "SQL": "Learn SQL through Mode Analytics, W3Schools, or SQLZoo. Practice with real datasets on platforms like Kaggle.",
        "R": "Take R courses on Coursera or DataCamp. Practice statistical analysis and data visualization with ggplot2.",
        "Go": "Learn Go through the official Go tour and build concurrent applications. Practice on Exercism or HackerRank.",
        
        # Web Technologies
        "React": "Follow React official documentation and build projects like a todo app, dashboard, or e-commerce site. Learn React hooks and state management.",
        "Angular": "Complete Angular tutorials on angular.io and build single-page applications. Learn TypeScript and RxJS.",
        "Vue.js": "Learn Vue.js through vuejs.org and build progressive web applications. Practice with Vue CLI and Vuex.",
        "Node.js": "Build RESTful APIs and real-time applications with Node.js. Learn Express.js and database integration.",
        "Django": "Build web applications with Django. Learn Django REST framework for API development.",
        "Flask": "Create lightweight web applications with Flask. Learn about microservices architecture.",
        "Spring": "Learn Spring Framework through Spring.io documentation. Build enterprise Java applications.",
        
        # Databases
        "MongoDB": "Learn MongoDB through MongoDB University courses. Practice document modeling and aggregation pipelines.",
        "PostgreSQL": "Study PostgreSQL advanced features like JSON support, full-text search, and window functions.",
        "MySQL": "Master MySQL through hands-on practice. Learn about indexing, optimization, and replication.",
        "Redis": "Learn Redis data structures and use cases. Practice caching strategies and pub/sub messaging.",
        
        # Cloud & DevOps
        "Docker": "Complete Docker tutorials and practice containerization projects. Learn multi-container applications with Docker Compose.",
        "Kubernetes": "Learn Kubernetes through official documentation and hands-on labs. Practice with minikube or cloud providers.",
        "AWS": "Obtain AWS certifications starting with Cloud Practitioner. Practice with AWS Free Tier services.",
        "Azure": "Get Azure fundamentals certification and practice with Azure services. Learn about Azure DevOps.",
        "GCP": "Learn Google Cloud Platform through Qwiklabs and obtain GCP Associate Cloud Engineer certification.",
        "Terraform": "Study Infrastructure as Code with Terraform. Practice with multiple cloud providers.",
        
        # Machine Learning & AI
        "Machine Learning": "Take Andrew Ng's Machine Learning course on Coursera. Practice with scikit-learn and TensorFlow.",
        "Deep Learning": "Study deep learning through deeplearning.ai courses. Practice with PyTorch and computer vision projects.",
        "TensorFlow": "Complete TensorFlow tutorials and build neural networks. Learn about TensorFlow Serving and TF Lite.",
        "PyTorch": "Learn PyTorch through official tutorials. Build projects in computer vision and NLP.",
        "NLP": "Study natural language processing with NLTK and spaCy. Build chatbots and sentiment analysis projects.",
        "Computer Vision": "Learn computer vision with OpenCV and TensorFlow. Build image classification and object detection projects.",
        
        # Data & Analytics
        "Data Science": "Enroll in Data Science specialization on Coursera or edX. Practice with pandas, NumPy, and scikit-learn.",
        "Data Analysis": "Learn data analysis with pandas and NumPy. Practice with real datasets on Kaggle.",
        "Tableau": "Take Tableau courses and create interactive dashboards. Learn about calculated fields and parameters.",
        "Power BI": "Learn Power BI through Microsoft Learn. Practice with DAX formulas and data modeling.",
        
        # Tools & Platforms
        "Git": "Master Git through interactive tutorials on Learn Git Branching. Practice branching strategies and collaboration workflows.",
        "Jenkins": "Learn CI/CD with Jenkins. Practice pipeline as code with Jenkinsfile.",
        "Agile": "Get certified in Agile methodologies through Scrum.org or PMI. Read 'Scrum Guide' and 'Agile Manifesto'.",
        
        # Security
        "Cybersecurity": "Study cybersecurity through CompTIA Security+ or CISSP courses. Practice with Capture The Flag (CTF) challenges.",
        "Penetration Testing": "Learn penetration testing through Offensive Security courses. Practice with Kali Linux and Metasploit.",
        
        # Soft Skills
        "Communication": "Join Toastmasters International to improve public speaking and communication. Practice writing technical documentation.",
        "Leadership": "Read leadership books like 'The 7 Habits of Highly Effective People' and consider taking management courses.",
        "Problem Solving": "Practice problem-solving with coding challenges on LeetCode, HackerRank, or Project Euler.",
        "Teamwork": "Participate in open-source projects or hackathons to improve collaboration skills.",
        "Project Management": "Get PMP or PRINCE2 certification. Practice with project management tools like Jira or Trello.",
        "Critical Thinking": "Take critical thinking courses and practice analyzing case studies and business problems."
    }
    
    # Role-specific recommendations
    role_recommendations = {
        "Data Analyst": [
            "Focus on SQL, Python/R, and data visualization tools like Tableau or Power BI.",
            "Learn statistical analysis and hypothesis testing through practical projects.",
            "Practice with real-world datasets from Kaggle or government open data portals."
        ],
        "Software Engineer": [
            "Master at least one programming language deeply and learn modern frameworks.",
            "Practice system design and coding interviews regularly.",
            "Contribute to open-source projects to gain real-world development experience."
        ],
        "ML Engineer": [
            "Deepen your knowledge of machine learning algorithms and frameworks.",
            "Practice with end-to-end ML projects from data preprocessing to deployment.",
            "Learn MLOps tools like MLflow, Kubeflow, and model serving platforms."
        ],
        "Full Stack Developer": [
            "Develop expertise in both frontend and backend technologies.",
            "Build full-stack applications with modern frameworks and databases.",
            "Learn about cloud deployment and DevOps practices."
        ],
        "DevOps Engineer": [
            "Master containerization with Docker and orchestration with Kubernetes.",
            "Learn Infrastructure as Code tools like Terraform and configuration management.",
            "Practice with CI/CD pipelines and monitoring solutions."
        ],
        "Product Manager": [
            "Learn product discovery and requirement gathering techniques.",
            "Practice with product metrics and user research methodologies.",
            "Study successful product case studies and business strategy frameworks."
        ],
        "UI/UX Designer": [
            "Master design tools like Figma, Sketch, or Adobe XD.",
            "Practice user research and usability testing methods.",
            "Build a portfolio showcasing design thinking and problem-solving skills."
        ],
        "Cybersecurity Analyst": [
            "Study security frameworks and compliance standards.",
            "Practice with security tools and penetration testing methodologies.",
            "Stay updated with the latest threats and security trends."
        ]
    }
    
    # Add role-specific recommendations first if role is provided
    if role and role in role_recommendations:
        recommendations.extend(role_recommendations[role])
    
    # Add skill-specific recommendations
    for skill in missing_skills[:15]:  # Limit to top 15 recommendations
        skill_key = skill.split()[0].title()  # Use first word for mapping
        if skill_key in recommendation_map:
            recommendations.append(recommendation_map[skill_key])
        else:
            # Try to find a partial match
            found = False
            for key in recommendation_map:
                if skill_key.lower() in key.lower():
                    recommendations.append(recommendation_map[key])
                    found = True
                    break
            if not found:
                recommendations.append(f"Research and develop proficiency in {skill} through online courses, documentation, and hands-on practice.")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    return unique_recommendations[:20]  # Limit to 20 recommendations

# Role selection
st.subheader("Select Target Role")
st.info("Choose the role you want to analyze your skills against")

# Create role cards for better UX
role_names = list(ROLE_SKILLS.keys())
selected_role = st.selectbox("Select Role", role_names)

# Show role skills preview
if selected_role:
    tech_skills, soft_skills = get_role_skills(selected_role)
    with st.expander(f"Required Skills for {selected_role}", expanded=False):
        st.markdown("**Technical Skills:**")
        for skill in tech_skills:
            st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
        st.markdown("**Soft Skills:**")
        for skill in soft_skills:
            st.markdown(f'<span class="skill-tag soft-skill">{skill}</span>', unsafe_allow_html=True)

# File upload section
st.subheader("Upload Your Resume")
resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], key="resume_uploader")

# Analyze button
analyze_button = st.button("Analyze Skills", type="primary")

if analyze_button:
    if not resume_file:
        st.warning("Please upload your resume.")
    elif not selected_role:
        st.warning("Please select a target role.")
    else:
        with st.spinner("Analyzing your skills..."):
            # Extract text from resume
            try:
                # Process resume
                if resume_file.name.endswith(".pdf"):
                    resume_text = extract_text_from_pdf(resume_file)
                elif resume_file.name.endswith(".docx"):
                    resume_text = extract_text_from_docx(resume_file)
                else:  # txt
                    resume_text = extract_text_from_txt(resume_file)
                
                # Extract skills from resume
                resume_tech_skills, resume_soft_skills = extract_skills(resume_text)
                
                # Get skills for selected role
                jd_tech_skills, jd_soft_skills = get_role_skills(selected_role)
                
                # Combine skills
                resume_all_skills = resume_tech_skills + resume_soft_skills
                jd_all_skills = jd_tech_skills + jd_soft_skills
                
                # Store combined skills in results for later use
                st.session_state.resume_all_skills = resume_all_skills
                st.session_state.jd_all_skills = jd_all_skills
                st.session_state.selected_role = selected_role
                
                # Compute similarity
                overall_match = compute_similarity(resume_all_skills, jd_all_skills)
                
                # Categorize skills
                matched_skills, partial_skills, missing_skills = categorize_skills(
                    resume_all_skills, jd_all_skills
                )
                
                # Generate recommendations
                recommendations = get_recommendations(missing_skills, selected_role)
                
                # Store results in session state
                st.session_state.results = {
                    "resume_tech_skills": resume_tech_skills,
                    "resume_soft_skills": resume_soft_skills,
                    "jd_tech_skills": jd_tech_skills,
                    "jd_soft_skills": jd_soft_skills,
                    "overall_match": overall_match,
                    "matched_skills": matched_skills,
                    "partial_skills": partial_skills,
                    "missing_skills": missing_skills,
                    "recommendations": recommendations
                }
                
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")

# Display results if available
if "results" in st.session_state:
    results = st.session_state.results
    
    # Section 2: Extracted Skills Display
    st.markdown('<h3 class="section-header">Extracted Skills</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("Your Resume Skills", expanded=True):
            st.subheader("Technical Skills")
            for skill in results["resume_tech_skills"]:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            
            st.subheader("Soft Skills")
            for skill in results["resume_soft_skills"]:
                st.markdown(f'<span class="skill-tag soft-skill">{skill}</span>', unsafe_allow_html=True)
    
    with col2:
        with st.expander(f"Required Skills for {st.session_state.selected_role}", expanded=True):
            st.subheader("Technical Skills")
            for skill in results["jd_tech_skills"]:
                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            
            st.subheader("Soft Skills")
            for skill in results["jd_soft_skills"]:
                st.markdown(f'<span class="skill-tag soft-skill">{skill}</span>', unsafe_allow_html=True)
    
    # Section 3: Skill Gap Analysis Results
    st.markdown('<h3 class="section-header">Skill Gap Analysis Results</h3>', unsafe_allow_html=True)
    
    # Overall Match Percentage
    st.metric("Overall Match Percentage", f"{results['overall_match']:.1f}%")
    
    # Progress bar
    st.progress(results['overall_match'] / 100)
    
    # Skill Match Overview (Donut chart)
    match_counts = {
        "Matched": len(results["matched_skills"]),
        "Partial Match": len(results["partial_skills"]),
        "Missing": len(results["missing_skills"])
    }
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=list(match_counts.keys()),
        values=list(match_counts.values()),
        hole=.5,
        marker_colors=['#10B981', '#F59E0B', '#EF4444']
    )])
    fig_donut.update_layout(title="Skill Match Overview")
    st.plotly_chart(fig_donut)
    
    # Missing Skills List
    st.subheader("Missing Skills")
    if results["missing_skills"]:
        for skill in results["missing_skills"]:
            st.markdown(f'<span class="skill-tag missing-skill">{skill}</span>', unsafe_allow_html=True)
    else:
        st.success("No missing skills! Great match!")
    
    # Similarity Matrix Heatmap
    if 'resume_all_skills' in st.session_state and 'jd_all_skills' in st.session_state:
        if st.session_state.resume_all_skills and st.session_state.jd_all_skills:
            # Create a similarity matrix
            resume_embeddings = model.encode(st.session_state.resume_all_skills)
            jd_embeddings = model.encode(st.session_state.jd_all_skills)
            similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings)
            
            # Create heatmap
            fig_heatmap = px.imshow(
                similarity_matrix,
                labels=dict(x=f"{st.session_state.selected_role} Skills", y="Your Skills", color="Similarity"),
                x=st.session_state.jd_all_skills,
                y=st.session_state.resume_all_skills,
                color_continuous_scale="RdYlGn",
                aspect="auto"
            )
            fig_heatmap.update_layout(title="Skills Similarity Heatmap", height=500)
            st.plotly_chart(fig_heatmap)
        else:
            st.info("Not enough skills extracted to generate similarity heatmap.")
    else:
        st.info("Please analyze documents to generate similarity heatmap.")
    
    # Section 4: Recommendations
    st.markdown('<h3 class="section-header">Upskilling Recommendations</h3>', unsafe_allow_html=True)
    
    if results["recommendations"]:
        for i, rec in enumerate(results["recommendations"], 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("No specific recommendations at this time.")
    
    # Role-specific recommendations
    st.subheader("Role-Specific Learning Path")
    st.info(f"Based on the {st.session_state.selected_role} role, we recommend focusing on the missing skills above to improve your match percentage.")
    
    # Section 5: Export Options
    st.markdown('<h3 class="section-header">Export Results</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as CSV"):
            # Create DataFrame for export
            df_data = {
                "Category": ["Matched Skills"] * len(results["matched_skills"]) +
                           ["Missing Skills"] * len(results["missing_skills"]),
                "Skill": results["matched_skills"] + results["missing_skills"]
            }
            df = pd.DataFrame(df_data)
            
            # Convert to CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{st.session_state.selected_role}_skill_gap_analysis.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("Export as PDF"):
            st.info("PDF export functionality would be implemented here.")

# Initialize session state
if "results" not in st.session_state:
    st.session_state.results = None
if "resume_all_skills" not in st.session_state:
    st.session_state.resume_all_skills = []
if "jd_all_skills" not in st.session_state:
    st.session_state.jd_all_skills = []
if "selected_role" not in st.session_state:
    st.session_state.selected_role = ""