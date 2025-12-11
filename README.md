# Intelligent Resume Ranker
<p> A machine learning powered resume evaluation system that ranks resumes based on semantic similarity to a job description. The system uses transformer embeddings, keyword extraction, and similarity scoring to produce accurate rankings and detailed skill-gap analysis. </p>
## Overview

The Intelligent Resume Ranker allows users to:

Upload multiple resumes (PDF, DOCX, TXT)

Compare them against a job description

Generate semantic similarity scores

Extract top keywords from the job description

Identify missing skills for each candidate

View ranked results with detailed insights

Export results as a CSV file

The application is built using Streamlit and Sentence Transformers.

## Features

Semantic resume-to-JD comparison

Transformer-based embeddings

Keyword extraction and gap identification

Detailed per-resume analysis

Supports PDF, DOCX, and TXT formats

CSV export for ranking results

Interactive web-based interface

## Project Structure
├── app.py                   # Streamlit user interface  
├── resume_ranker.py         # Core resume ranking logic  
├── requirements.txt         # Project dependencies  
├── images/                  # Screenshots  
├── README.md                # Documentation  
└── sample_resumes/          # Example resumes  

## Installation
1. Clone the repository
git clone https://github.com/<your-username>/resume-ranker.git
cd resume-ranker

2. Create a virtual environment
python -m venv .venv

3. Activate the environment

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

4. Install dependencies
pip install -r requirements.txt

Running the Application
streamlit run app.py


The app will be available at:

http://localhost:8501

## How It Works
1. Embedding Generation

Resumes and the job description are converted into dense vector embeddings using a Sentence Transformers model.

2. Similarity Computation

Cosine similarity is used to measure how closely each resume aligns with the job description.

3. Keyword Extraction

Top keywords from the JD are extracted using TF-IDF and semantic grouping.

4. Skill Gap Detection

Missing skills or keywords are identified per resume.

5. Ranking

Resumes are ranked by similarity score and presented with detailed metrics.

## Screenshots

Add your screenshot paths below:

<p align="center">
  <img src="images/screenshot1.png" width="650">
</p>

<p align="center">
  <img src="images/screenshot2.png" width="650">
</p>

## CSV Output Format

The exported CSV contains:

filename	score	missing_skills_count	missing_skills
## Models Used

Sentence Transformers (default: all-MiniLM-L6-v2)

Optional: all-mpnet-base-v2

FAISS (optional backend for vector indexing)

Contributing

Suggestions and improvements are welcome.
## Potential enhancements include:

Support for multilingual resumes

Improved PDF text extraction

LLM-powered resume improvement suggestions

Integration with an applicant tracking system (ATS)

## Author

Sheeba Nadeem
GitHub: https://github.com/sheebanadeem
