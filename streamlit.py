import streamlit as st
import re
import nltk
import pickle
import PyPDF2
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

with open("randomforest.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

#Skill keywords/ job role requirements
skill_keywords = ['python', 'machine learning', 'sql', 'nlp', 'java', 'c++', 'pandas', 'keras']
job_roles = {
    "Data Scientist": {"python", "machine learning", "sql", "pandas", "deep learning"},
    "Software Engineer": {"java", "c++", "sql", "data structures", "algorithms"},
    "Business Analyst": {"excel", "sql", "data visualization", "power bi"},
}

#Resouroces
learning_resources = {
    "sql": ["W3Schools SQL Tutorial: https://www.w3schools.com/sql/",
            "SQL for Data Science (Coursera): https://www.coursera.org/learn/sql-for-data-science"],
    
    "machine learning": ["Machine Learning Crash Course by Google: https://developers.google.com/machine-learning/crash-course",
                          "Hands-On Machine Learning with Scikit-Learn (Book): https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/"],
    
    "python": ["Python for Beginners (W3Schools): https://www.w3schools.com/python/",
               "Python Data Science Handbook: https://jakevdp.github.io/PythonDataScienceHandbook/"],
}

#clean resume text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

#skills extaction from resume text
def extract_skills(text):
    return [skill for skill in skill_keywords if skill in text]

#find missing skills based on job role
def missing_skills(resume_skills, role):
    required_skills = job_roles.get(role, set())
    return required_skills - set(resume_skills)

#suggest learning resources for missing skills
def suggest_resources(missing_skills):
    resources = {}
    for skill in missing_skills:
        if skill in learning_resources:
            resources[skill] = learning_resources[skill]
    return resources

#transform resume text into model features
def extract_features(text):
    text_tfidf = vectorizer.transform([text])                       #text to feature vector
    return text_tfidf

#predict job category using trained model
def predict_category(text):
    features = extract_features(text)                               #text to features
    prediction = model.predict(features)                            #model prediction
    return prediction[0]                                            # Returning the predicted label

#text extraction from uploaded file
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    text = ""

    if file_type == "text/plain":
        try:
            text = uploaded_file.read().decode("utf-8", errors="ignore")  # Ignore decoding errors
        except UnicodeDecodeError:
            st.error("⚠️ Error decoding the text file. Please upload a properly encoded UTF-8 file.")

    elif file_type == "application/pdf":
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"⚠️ Error reading PDF: {e}")

    return text

#Streamlit UI
st.title("📄 NLP-Based Resume Analysis & Adaptive Skill Assessment")

#1:Resume Upload
uploaded_file = st.file_uploader("Upload your resume (TXT or PDF)", type=["txt", "pdf"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    
    # Reading the file
    resume_text = extract_text_from_file(uploaded_file)
    
    if resume_text:
        #2: Clean the resume text
        cleaned_text = clean_text(resume_text)
        st.write("### 🔹 Cleaned Resume Text:")
        st.write(cleaned_text)

          
        #3: Predict resume category using ML model
        prediction = predict_category(cleaned_text)
        st.markdown("---")
        st.write("### 🏆 Predicted Job Category:")
        # Display classification result
        st.write(f"📌 {prediction}")
        st.markdown("---")  
    

        #4: Extract skills
        extracted_skills = extract_skills(cleaned_text)
        st.write("### 🔹 Extracted Skills:")
        st.write(extracted_skills)

        #5: User selects job role
        job_role = st.selectbox("📌 Select a job role:", list(job_roles.keys()))

        #6: Find missing skills
        missing = missing_skills(extracted_skills, job_role)
        st.write("### ❌ Missing Skills:")
        st.write(missing)

        #7: Suggest learning resources for missing skills
        resources = suggest_resources(missing)
        st.write("### 📚 Suggested Learning Resources:")
        for skill, links in resources.items():
            st.write(f"**{skill.capitalize()} Resources:**")
            for link in links:
                st.markdown(f"- [🔗 {link}]({link})")
