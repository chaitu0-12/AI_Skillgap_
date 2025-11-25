import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [roles, setRoles] = useState([]);
  const [selectedRole, setSelectedRole] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Fetch available roles on component mount
  useEffect(() => {
    fetchRoles();
  }, []);

  const fetchRoles = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/roles');
      setRoles(response.data.roles);
      if (response.data.roles.length > 0) {
        setSelectedRole(response.data.roles[0]);
      }
    } catch (err) {
      setError('Failed to fetch roles');
    }
  };

  const handleFileChange = (e) => {
    setResumeFile(e.target.files[0]);
  };

  const handleRoleChange = (e) => {
    setSelectedRole(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!resumeFile) {
      setError('Please upload a resume');
      return;
    }
    if (!selectedRole) {
      setError('Please select a role');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('resume', resumeFile);
    formData.append('role', selectedRole);

    try {
      const response = await axios.post('http://localhost:5000/api/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      setAnalysisResults(response.data);
    } catch (err) {
      setError('Failed to analyze skills: ' + (err.response?.data?.error || err.message));
    } finally {
      setLoading(false);
    }
  };

  const exportResults = async () => {
    if (!analysisResults) return;

    try {
      const exportData = {
        ...analysisResults,
        role: selectedRole
      };
      
      const response = await axios.post('http://localhost:5000/api/export', exportData, {
        responseType: 'blob'
      });
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', response.headers['content-disposition']?.split('filename=')[1] || 'skill_gap_analysis.csv');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Failed to export results');
    }
  };

  // Chart data preparation
  const getSkillMatchChartData = () => {
    if (!analysisResults) return null;
    
    return {
      labels: ['Matched', 'Partial Match', 'Missing'],
      datasets: [
        {
          label: 'Skills',
          data: [
            analysisResults.matched_skills.length,
            analysisResults.partial_skills.length,
            analysisResults.missing_skills.length
          ],
          backgroundColor: [
            '#10B981', // Green for matched
            '#F59E0B', // Amber for partial
            '#EF4444'  // Red for missing
          ],
          borderColor: [
            '#047857',
            '#B45309',
            '#B91C1C'
          ],
          borderWidth: 1
        }
      ]
    };
  };

  const getOverallMatchData = () => {
    if (!analysisResults) return null;
    
    // Calculate the complement to make a proper doughnut chart
    const matchPercentage = analysisResults.overall_match;
    const remainingPercentage = 100 - matchPercentage;
    
    return {
      labels: ['Match', 'Remaining'],
      datasets: [
        {
          label: 'Overall Match Percentage',
          data: [matchPercentage, remainingPercentage],
          backgroundColor: ['#4F46E5', '#E5E7EB'],
          borderColor: ['#4338CA', '#D1D5DB'],
          borderWidth: 1
        }
      ]
    };
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>
            AI Role-Based Skill Gap Analyzer
          </h1>
          <p>
            Upload your resume and select your target role to identify skill gaps and get upskilling recommendations.
          </p>
        </div>
      </header>

      <main className="main-content">
        {/* Error message */}
        {error && (
          <div className="alert alert-error">
            <div className="alert-icon">⚠️</div>
            <div className="alert-content">{error}</div>
            <button className="alert-close" onClick={() => setError('')}>×</button>
          </div>
        )}

        {/* Analysis Form */}
        <div className="card">
          <h2 className="card-title">Analyze Your Skills</h2>
          <form onSubmit={handleSubmit}>
            <div>
              {/* Role Selection */}
              <div className="form-group">
                <label htmlFor="role" className="form-label">
                  Select Target Role
                </label>
                <select
                  id="role"
                  value={selectedRole}
                  onChange={handleRoleChange}
                  className="form-select"
                  disabled={loading}
                >
                  {roles.map((role) => (
                    <option key={role} value={role}>
                      {role}
                    </option>
                  ))}
                </select>
              </div>

              {/* Resume Upload */}
              <div className="form-group">
                <label htmlFor="resume" className="form-label">
                  Upload Resume (PDF, DOCX, TXT)
                </label>
                <div className="file-upload-wrapper">
                  <input
                    type="file"
                    id="resume"
                    accept=".pdf,.docx,.txt"
                    onChange={handleFileChange}
                    className="form-input file-input"
                    disabled={loading}
                  />
                  <div className="file-upload-placeholder">
                    {resumeFile ? resumeFile.name : 'Choose a file or drag it here'}
                  </div>
                </div>
                <div className="file-upload-hint">
                  Supported formats: PDF, DOCX, TXT (Max size: 5MB)
                </div>
              </div>

              {/* Submit Button */}
              <div className="form-actions">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner-small"></span>
                      Analyzing...
                    </>
                  ) : (
                    'Analyze Skills'
                  )}
                </button>
              </div>
            </div>
          </form>
        </div>

        {/* Loading Indicator */}
        {loading && (
          <div className="loading-overlay">
            <div className="loading-card">
              <div className="spinner"></div>
              <h3>Analyzing Your Resume</h3>
              <p>Comparing your skills with {selectedRole} requirements...</p>
              <div className="loading-progress">
                <div className="loading-progress-bar"></div>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {analysisResults && !loading && (
          <div className="results-section">
            <div className="card">
              <div className="results-header">
                <h2 className="card-title">Analysis Results for {selectedRole}</h2>
                <div className="results-actions">
                  <button
                    onClick={exportResults}
                    className="btn btn-secondary"
                  >
                    Export Results
                  </button>
                </div>
              </div>

              {/* Overall Match */}
              <div className="result-card highlight-card">
                <h3>Overall Match</h3>
                <div className="match-score">
                  <div className="match-score-value">{analysisResults.overall_match.toFixed(1)}%</div>
                  <div className="match-score-label">Match with {selectedRole} requirements</div>
                </div>
                <div className="progress-bar">
                  <div 
                    className={`progress-fill progress-high`} 
                    style={{ width: `${analysisResults.overall_match}%` }}
                  ></div>
                </div>
              </div>

              {/* Charts */}
              <div className="chart-grid">
                <div className="chart-container">
                  <h3>Skill Match Distribution</h3>
                  <Bar data={getSkillMatchChartData()} />
                </div>

                <div className="chart-container">
                  <h3>Overall Match Percentage</h3>
                  <Doughnut data={getOverallMatchData()} />
                </div>
              </div>

              {/* Detailed Results */}
              <div className="results-grid">
                {/* Matched Skills */}
                <div className="result-card">
                  <div className="skill-category-header matched">
                    <h3>
                      <span className="match-indicator match-high"></span>
                      Matched Skills
                    </h3>
                    <span className="skill-count">{analysisResults.matched_skills.length}</span>
                  </div>
                  <ul className="skill-list">
                    {analysisResults.matched_skills.map((skill, index) => (
                      <li key={index} className="skill-item">
                        <span className="skill-name">{skill}</span>
                        <span className="skill-status matched">✓</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Partial Match Skills */}
                <div className="result-card">
                  <div className="skill-category-header partial">
                    <h3>
                      <span className="match-indicator match-medium"></span>
                      Partial Match Skills
                    </h3>
                    <span className="skill-count">{analysisResults.partial_skills.length}</span>
                  </div>
                  <ul className="skill-list">
                    {analysisResults.partial_skills.map((skill, index) => (
                      <li key={index} className="skill-item">
                        <span className="skill-name">{skill}</span>
                        <span className="skill-status partial">~</span>
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Missing Skills */}
                <div className="result-card">
                  <div className="skill-category-header missing">
                    <h3>
                      <span className="match-indicator match-low"></span>
                      Missing Skills
                    </h3>
                    <span className="skill-count">{Math.min(analysisResults.missing_skills.length, 10)}</span>
                  </div>
                  <ul className="skill-list">
                    {analysisResults.missing_skills.slice(0, 10).map((skill, index) => (
                      <li key={index} className="skill-item">
                        <span className="skill-name">{skill}</span>
                        <span className="skill-status missing">✗</span>
                      </li>
                    ))}
                  </ul>
                  {analysisResults.missing_skills.length > 10 && (
                    <div className="more-skills">
                      + {analysisResults.missing_skills.length - 10} more skills not shown
                    </div>
                  )}
                </div>
              </div>

              {/* Recommendations */}
              <div className="result-card recommendations-card">
                <h3>Upskilling Recommendations</h3>
                <div className="recommendations-list">
                  {analysisResults.recommendations.slice(0, 10).map((rec, index) => (
                    <div key={index} className="recommendation-item">
                      <div className="recommendation-number">#{index + 1}</div>
                      <div className="recommendation-content">{rec}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <p>AI Role-Based Skill Gap Analyzer &copy; 2025</p>
          <p className="footer-links">
            <a href="#" onClick={(e) => { e.preventDefault(); alert('Help documentation coming soon!'); }}>Help</a>
            <span className="footer-separator">|</span>
            <a href="#" onClick={(e) => { e.preventDefault(); alert('Privacy policy coming soon!'); }}>Privacy</a>
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;