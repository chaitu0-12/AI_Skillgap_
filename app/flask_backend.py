import os
import io
import json
import pandas as pd
import numpy as np
import pdfplumber
import docx
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import re
from fuzzywuzzy import fuzz, process
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face authentication
try:
    from app.huggingface_auth import is_huggingface_authenticated, authenticate_huggingface
except ImportError:
    from huggingface_auth import is_huggingface_authenticated, authenticate_huggingface

# Check Hugging Face authentication
if not is_huggingface_authenticated():
    print("Warning: Hugging Face token not found. Please set your HUGGINGFACE_TOKEN in the .env file for full functionality.")
else:
    try:
        token = authenticate_huggingface()
        print("Successfully authenticated with Hugging Face")
    except Exception as e:
        print(f"Hugging Face authentication failed: {str(e)}")

# Import role skills data
try:
    from app.roles_data import ROLE_SKILLS
except ImportError:
    from roles_data import ROLE_SKILLS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Please install the spaCy English model: python -m spacy download en_core_web_sm")
    exit(1)

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(file_stream):
    """Extract text from PDF file"""
    text = ""
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_stream):
    """Extract text from DOCX file"""
    doc = docx.Document(file_stream)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(file_stream):
    """Extract text from TXT file"""
    return file_stream.read().decode("utf-8")

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
        
        # Machine Learning & AI - Expanded with more specific terms
        "tensorflow", "pytorch", "scikit-learn", "scikit learn", "keras", "pandas", "numpy", "matplotlib",
        "seaborn", "plotly", "bokeh", "jupyter", "jupyter notebook", "machine learning", "ml",
        "deep learning", "neural networks", "nlp", "natural language processing",
        "computer vision", "cv", "reinforcement learning", "rl", "data science", "data analysis",
        "data engineering", "big data", "hadoop", "spark", "kafka", "flink", "storm",
        "airflow", "mlflow", "kubeflow", "fastapi", "streamlit", "gradio", "hugging face",
        "transformers", "bert", "gpt", "llm", "large language model", "chatgpt", "openai",
        "xgboost", "lightgbm", "catboost", "random forest", "svm", "support vector machine",
        "decision trees", "ensemble methods", "hyperparameter tuning", "cross validation",
        "model evaluation", "confusion matrix", "roc curve", "auc", "precision", "recall", 
        "f1 score", "overfitting", "underfitting", "regularization", "feature engineering",
        "dimensionality reduction", "pca", "tsne", "clustering", "k-means", "dbscan",
        "time series analysis", "arima", "lstm", "gru", "gans", "autoencoders",
        "transfer learning", "fine-tuning", "model compression", "quantization",
        "pruning", "distillation", "model serving", "tf serving", "torchserve",
        "onnx", "tensorrt", "model monitoring", "drift detection", "bias detection",
        "mlops", "model deployment", "feature store", "experiment tracking",
        "model registry", "pipeline orchestration", "model versioning",
        
        # Data & Analytics - Expanded with more specific terms
        "excel", "tableau", "power bi", "looker", "qlik", "qliksense", "sisense", "mode",
        "metabase", "redash", "superset", "analytics", "business intelligence", "bi",
        "statistics", "statistical", "r", "sas", "spss", "matlab", "stata", "data mining",
        "data visualization", "dashboard", "kpi", "metrics", "reporting", "ab testing",
        "a/b testing", "hypothesis testing", "regression", "clustering", "classification",
        "descriptive statistics", "inferential statistics", "correlation analysis",
        "anova", "chi-square test", "t-test", "z-test", "statistical modeling",
        
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
    
    # Split text into paragraphs and sentences for better context detection
    paragraphs = text.split('\n\n')
    sentences = [sent.strip() for para in paragraphs for sent in para.split('.') if sent.strip()]
    
    # Extract named entities
    entities = [ent.text.lower() for ent in doc.ents]
    
    # Extract noun chunks
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    
    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    # Combine all text elements with context
    all_text_elements = entities + noun_chunks + text.lower().split() + lemmatized_tokens
    
    # Add context-aware elements
    context_elements = []
    for para in paragraphs:
        context_elements.extend(para.lower().split())
    for sent in sentences:
        context_elements.extend(sent.lower().split())
    
    all_text_elements.extend(context_elements)
    
    # Find technical skills with improved fuzzy matching
    technical_skills = []
    for skill in technical_skills_keywords:
        # Exact match
        if any(skill in element for element in all_text_elements):
            technical_skills.append(skill.title())
        # Partial match for compound skills
        elif any(len(skill.split()) > 1 and all(word in element for word in skill.split()) for element in all_text_elements):
            technical_skills.append(skill.title())
        # Improved fuzzy match for similar skills (threshold reduced to 60 for better detection)
        else:
            matches = process.extract(skill, all_text_elements, limit=5)
            for match in matches:
                if match[1] >= 60:  # Lower threshold for better detection
                    technical_skills.append(skill.title())
                    break
    
    # Find soft skills with improved fuzzy matching
    soft_skills = []
    for skill in soft_skills_keywords:
        # Exact match
        if any(skill in element for element in all_text_elements):
            soft_skills.append(skill.title())
        # Partial match for compound skills
        elif any(len(skill.split()) > 1 and all(word in element for word in skill.split()) for element in all_text_elements):
            soft_skills.append(skill.title())
        # Improved fuzzy match for similar skills (threshold reduced to 60 for better detection)
        else:
            matches = process.extract(skill, all_text_elements, limit=5)
            for match in matches:
                if match[1] >= 60:  # Lower threshold for better detection
                    soft_skills.append(skill.title())
                    break
    
    # Remove duplicates and sort
    technical_skills = sorted(list(set(technical_skills)))
    soft_skills = sorted(list(set(soft_skills)))
    
    # Increase limit to 25 technical and 15 soft skills for better accuracy
    technical_skills = technical_skills[:25]
    soft_skills = soft_skills[:15]
    
    return technical_skills, soft_skills

def get_role_skills(role):
    """Get skills for a specific role"""
    if role in ROLE_SKILLS:
        return ROLE_SKILLS[role]["technical_skills"], ROLE_SKILLS[role]["soft_skills"]
    return [], []

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
    
    # Limit missing skills to 10 items
    missing_skills = missing_skills[:10]
    
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
    
    # Add skill-specific recommendations for top 5 missing skills only
    for skill in missing_skills[:5]:  # Limit to top 5 recommendations
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
    
    return unique_recommendations[:10]  # Limit to 10 recommendations

# API Routes
@app.route('/api/roles', methods=['GET'])
def get_roles():
    """Get list of available roles"""
    return jsonify({
        "roles": list(ROLE_SKILLS.keys())
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_skills():
    """Analyze skills from resume and role"""
    try:
        # Get role from request
        role = request.form.get('role')
        if not role:
            return jsonify({"error": "Role is required"}), 400
        
        # Get resume file
        if 'resume' not in request.files:
            return jsonify({"error": "Resume file is required"}), 400
        
        resume_file = request.files['resume']
        if resume_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Extract text from resume
        try:
            filename = resume_file.filename or ''
            if filename.endswith('.pdf'):
                resume_text = extract_text_from_pdf(resume_file)
            elif filename.endswith('.docx'):
                resume_text = extract_text_from_docx(resume_file)
            else:  # txt
                resume_text = extract_text_from_txt(resume_file)
        except Exception as e:
            return jsonify({"error": f"Error processing resume file: {str(e)}"}), 500
        
        # Extract skills from resume
        resume_tech_skills, resume_soft_skills = extract_skills(resume_text)
        
        # Get skills for selected role
        jd_tech_skills, jd_soft_skills = get_role_skills(role)
        
        # Combine skills
        resume_all_skills = resume_tech_skills + resume_soft_skills
        jd_all_skills = jd_tech_skills + jd_soft_skills
        
        # Compute similarity
        overall_match = compute_similarity(resume_all_skills, jd_all_skills)
        
        # Categorize skills
        matched_skills, partial_skills, missing_skills = categorize_skills(
            resume_all_skills, jd_all_skills
        )
        
        # Generate recommendations
        recommendations = get_recommendations(missing_skills, role)
        
        # Create similarity matrix for heatmap
        if resume_all_skills and jd_all_skills:
            resume_embeddings = model.encode(resume_all_skills)
            jd_embeddings = model.encode(jd_all_skills)
            similarity_matrix = cosine_similarity(resume_embeddings, jd_embeddings).tolist()
        else:
            similarity_matrix = []
        
        # Prepare response
        response = {
            "resume_tech_skills": resume_tech_skills,
            "resume_soft_skills": resume_soft_skills,
            "jd_tech_skills": jd_tech_skills,
            "jd_soft_skills": jd_soft_skills,
            "overall_match": round(overall_match, 2),
            "matched_skills": matched_skills,
            "partial_skills": partial_skills,
            "missing_skills": missing_skills,
            "recommendations": recommendations,
            "similarity_matrix": similarity_matrix,
            "resume_skills_labels": resume_all_skills,
            "jd_skills_labels": jd_all_skills
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export results as CSV"""
    try:
        data = request.get_json()
        
        # Create DataFrame for export
        df_data = {
            "Category": ["Matched Skills"] * len(data["matched_skills"]) +
                       ["Missing Skills"] * len(data["missing_skills"]),
            "Skill": data["matched_skills"] + data["missing_skills"]
        }
        df = pd.DataFrame(df_data)
        
        # Convert to CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        return jsonify({
            "csv": csv_content,
            "filename": f"{data['role']}_skill_gap_analysis.csv"
        })
    
    except Exception as e:
        return jsonify({"error": f"Error exporting results: {str(e)}"}), 500

# Serve React frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve React frontend"""
    if path != "" and os.path.exists(os.path.join(app.root_path, "frontend", "build", path)):
        return send_from_directory(os.path.join(app.root_path, "frontend", "build"), path)
    else:
        return send_from_directory(os.path.join(app.root_path, "frontend", "build"), 'index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)