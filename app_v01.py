import streamlit as st
import numpy as np
import re
import string
import spacy
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure the SpaCy model is installed
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

# Load the pre-trained model and tokenizer
model = load_model('bigru_model_n.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

# Define preprocessing and feature extraction functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == 'NOUN']
    return nouns

def find_matching_nouns(resume, job_description):
    resume_nouns = set(extract_nouns(resume))
    job_description_nouns = set(extract_nouns(job_description))
    matching_nouns = resume_nouns.intersection(job_description_nouns)
    return len(matching_nouns)

# Streamlit interface
st.title("ATS Score Predictor")
resume_input = st.text_area("Enter Resume Text")
jd_input = st.text_area("Enter Job Description")

if st.button("Predict ATS Score"):
    if resume_input and jd_input:
        # Preprocess the inputs
        cleaned_resume = clean_text(resume_input)
        cleaned_jd = clean_text(jd_input)

        # Tokenize and pad sequences
        max_len = 4000  # Ensure this is consistent with your model's expected input length
        resume_seq = tokenizer.texts_to_sequences([cleaned_resume])
        jd_seq = tokenizer.texts_to_sequences([cleaned_jd])
        resume_padded = pad_sequences(resume_seq, maxlen=max_len)
        jd_padded = pad_sequences(jd_seq, maxlen=max_len)

        # Combine the padded sequences
        combined_input = np.concatenate([resume_padded, jd_padded], axis=-1)

        # Extract noun feature
        matching_nouns = np.array([[find_matching_nouns(resume_input, jd_input)]])

        # Predict ATS score
        ats_score = model.predict([combined_input, matching_nouns])

        st.write(f"Predicted ATS Score: {ats_score[0][0]:.2f}")
    else:
        st.write("Please enter both resume text and job description.")
