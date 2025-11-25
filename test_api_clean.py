import requests
import json

# Test the API
url = "http://localhost:5000/api/analyze"
files = {"resume": open("d:\\infosys_resume_analyzer\\test_resume.txt", "rb")}
data = {"role": "Software Engineer"}

response = requests.post(url, files=files, data=data)
result = response.json()

# Print key information
print("Overall Match:", result.get("overall_match", "N/A"))
print("\nResume Technical Skills:")
for skill in result.get("resume_tech_skills", []):
    print(f"  - {skill}")

print("\nResume Soft Skills:")
for skill in result.get("resume_soft_skills", []):
    print(f"  - {skill}")

print("\nMissing Skills:")
for skill in result.get("missing_skills", [])[:10]:  # Show first 10
    print(f"  - {skill}")

print("\nRecommendations:")
for rec in result.get("recommendations", [])[:5]:  # Show first 5
    print(f"  - {rec}")