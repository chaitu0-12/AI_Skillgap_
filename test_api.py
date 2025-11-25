import requests

# Test the API
url = "http://localhost:5000/api/analyze"
files = {"resume": open("d:\\infosys_resume_analyzer\\test_resume.txt", "rb")}
data = {"role": "Software Engineer"}

response = requests.post(url, files=files, data=data)
print(response.json())