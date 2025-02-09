NLP-Based Resume Analysis & Adaptive Skill Assessment System

📌 Project Overview

This project aims to classify resumes into different categories and provide an adaptive skill assessment. It extracts relevant skills from resumes, compares them with job requirements, and suggests learning resources for missing skills.

🚀 Features

Resume Classification: Categorizes resumes using Decision Tree & Random Forest.

Skill Extraction: Uses NLP to extract skills from resumes.

Adaptive Skill Assessment: Identifies missing skills and suggests learning resources.

User-Friendly Interface: Deploys as a Streamlit web application.

🛠️ Tech Stack

Programming Language: Python

ML Models: Decision Tree, Random Forest

Libraries: Pandas, Scikit-learn, SpaCy, TfidfVectorizer

Deployment: Streamlit

📂 Dataset

Source: Kaggle (https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)

Fields: Resume Text, Job Categories

🔄 Workflow

Data Preprocessing:

Clean text (remove special characters, stopwords, lemmatization)

Convert text into numerical features using TF-IDF

Model Training & Evaluation:

Train Decision Tree & Random Forest

Evaluate using accuracy and classification report

Skill Extraction & Assessment:

Use NLP to extract skills from resumes

Compare with job skills & suggest missing ones

Deployment:

Create an interactive Streamlit app


🌟 How to Use

Install Dependencies:

pip install -r requirements.txt

Run the Streamlit App:

streamlit run app.py

Upload Resume & Get Insights!

🎯 Future Improvements

Add more advanced NLP techniques (BERT, Transformer-based models)

Expand dataset for better generalization

Improve skill recommendations with job market trends

🔗 Author: Your Name📧 Contact: kishorekumarbn18@gmail.com

