import pickle
import numpy as np
import spacy
import re
import string
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

abbreviations = {
    "mgr": "manager",
    "sr": "senior",
    "jr": "junior",
    "asst": "assistant",
    "assoc": "associate",
    "dept": "department",
    "exp": "experience",
    "hr": "human resources",
    "acct": "account",
    "acctg": "accounting",
    "fin": "finance",
    "eng": "engineer",
    "engg": "engineering",
    "it": "information technology",
    "qa": "quality assurance",
    "dev": "development",
    "devops": "development operations",
    "proj": "project",
    "mktg": "marketing",
    "biz": "business",
    "comm": "communication",
    "adm": "administration",
    "sec": "secretary",
    "exec": "executive",
    "corp": "corporation",
    "intl": "international",
    "rep": "representative",
    "mfg": "manufacturing",
    "prod": "production",
    "purch": "purchasing",
    "sales": "sales",
    "cust": "customer",
    "svc": "service",
    "tech": "technical",
    "sup": "supervisor",
    "supv": "supervision",
    "log": "logistics",
    "inv": "inventory",
    "sch": "schedule",
    "edu": "education",
    "lang": "language",
    "pr": "public relations",
    "hrd": "human resources development",
    "cfo": "chief financial officer",
    "ceo": "chief executive officer",
    "coo": "chief operating officer",
    "cmo": "chief marketing officer",
    "cto": "chief technology officer",
    "cio": "chief information officer",
    "pmo": "project management office",
    "pmp": "project management professional",
    "ba": "business analyst",
    "bpm": "business process management",
    "ui": "user interface",
    "ux": "user experience",
    "svp": "senior vice president",
    "vp": "vice president",
    "gm": "general manager",
    "doe": "depends on experience",
    "r&d": "research and development",
    "seo": "search engine optimization",
    "sem": "search engine marketing",
    "smm": "social media marketing",
    "b2b": "business to business",
    "b2c": "business to consumer",
    "kpi": "key performance indicator",
    "roi": "return on investment",
    "saas": "software as a service",
    "paas": "platform as a service",
    "iaas": "infrastructure as a service",
    "crm": "customer relationship management",
    "erp": "enterprise resource planning",
    "sd": "software development",
    "pm": "project manager",
    "pa": "personal assistant",
    "exec": "executive",
    "fin": "finance",
    "hrm": "human resources management",
    "it": "information technology",
    "pr": "public relations",
    "qa": "quality assurance",
    "r&d": "research and development",
    "scm": "supply chain management",
    "seo": "search engine optimization",
    "smm": "social media marketing",
    "ux": "user experience",
    "ui": "user interface",
    "bi": "business intelligence",
    "dev": "development",
    "ops": "operations"
}

def ensure_model_installed():
    try:
        spacy.load('en_core_web_sm')
    except OSError:
        from spacy.cli import download
        download('en_core_web_sm')
        spacy.load('en_core_web_sm')

# Ensure the model is installed
ensure_model_installed()

nlp = spacy.load("en_core_web_sm")

def expand_abbreviations(text, abbreviations):
    for abbr, expanded in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(abbr), expanded, text, flags=re.IGNORECASE)
    return text

def clean_and_preprocess(text):
    text = expand_abbreviations(text, abbreviations)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
    return nouns

def load_model_and_tokenizers_for_hr():
    model_path = 'modelfile/bighr2.keras'
    tokenizer_path = 'tokenizer/tokenizershr.pkl'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizers = pickle.load(f)
    resume_tokenizer = tokenizers.get('resume_tokenizer')
    description_tokenizer = tokenizers.get('description_tokenizer')
    common_nouns_tokenizer = tokenizers.get('common_nouns_tokenizer')
    if not (resume_tokenizer and description_tokenizer and common_nouns_tokenizer):
        raise ValueError("Tokenizer components are missing from the file.")
    return model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer

def load_model_and_tokenizers_for_it():
    model_path = 'modelfile/bigit2.keras'
    tokenizer_path = 'tokenizer/tokenizersit.pkl'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizers = pickle.load(f)
    resume_tokenizer = tokenizers.get('resume_tokenizer')
    description_tokenizer = tokenizers.get('description_tokenizer')
    common_nouns_tokenizer = tokenizers.get('common_nouns_tokenizer')
    if not (resume_tokenizer and description_tokenizer and common_nouns_tokenizer):
        raise ValueError("Tokenizer components are missing from the file.")
    return model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer

def load_model_and_tokenizers_for_sales():
    model_path = 'modelfile/bigrsales2.keras'
    tokenizer_path = 'tokenizer/tokenizerssales.pkl'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizers = pickle.load(f)
    resume_tokenizer = tokenizers.get('resume_tokenizer')
    description_tokenizer = tokenizers.get('description_tokenizer')
    common_nouns_tokenizer = tokenizers.get('common_nouns_tokenizer')
    if not (resume_tokenizer and description_tokenizer and common_nouns_tokenizer):
        raise ValueError("Tokenizer components are missing from the file.")
    return model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer

def load_model_and_tokenizers_for_health():
    model_path = 'modelfile/bighealth2.keras'
    tokenizer_path = 'tokernizer/tokenizershealth.pkl'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizers = pickle.load(f)
    resume_tokenizer = tokenizers.get('resume_tokenizer')
    description_tokenizer = tokenizers.get('description_tokenizer')
    common_nouns_tokenizer = tokenizers.get('common_nouns_tokenizer')
    if not (resume_tokenizer and description_tokenizer and common_nouns_tokenizer):
        raise ValueError("Tokenizer components are missing from the file.")
    return model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer

def load_model_and_tokenizers_for_other():
    model_path = 'modelfile/bigothers2.keras'
    tokenizer_path = 'tokernizer/tokenizersothers.pkl'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizers = pickle.load(f)
    resume_tokenizer = tokenizers.get('resume_tokenizer')
    description_tokenizer = tokenizers.get('description_tokenizer')
    common_nouns_tokenizer = tokenizers.get('common_nouns_tokenizer')
    if not (resume_tokenizer and description_tokenizer and common_nouns_tokenizer):
        raise ValueError("Tokenizer components are missing from the file.")
    return model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer

# Streamlit UI
st.title("ATS")

st.write("Upload your resume and job description, then select the job sector to analyze how well the resume fits the job description.")

# Resume input
resume = st.text_area("Paste your Resume:", height=150)

# Job description input
job_description = st.text_area("Paste Job Description:", height=150)

# Sector selection
sector = st.selectbox("Select Sector:", ['HR', 'IT', 'Sales', 'Health', 'Other'])

if st.button("Calculate ATS Score"):
    if resume and job_description:
        try:
            if sector == 'HR':
                model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer = load_model_and_tokenizers_for_hr()
            elif sector == 'IT':
                model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer = load_model_and_tokenizers_for_it()
            elif sector == 'Sales':
                model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer = load_model_and_tokenizers_for_sales()
            elif sector == 'Health':
                model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer = load_model_and_tokenizers_for_health()
            elif sector == 'Other':
                model, resume_tokenizer, description_tokenizer, common_nouns_tokenizer = load_model_and_tokenizers_for_other()

            processed_resume = clean_and_preprocess(resume)
            processed_description = clean_and_preprocess(job_description)

            resume_sequence = resume_tokenizer.texts_to_sequences([processed_resume])
            resume_data_padded = pad_sequences(resume_sequence, maxlen=1500)

            description_sequence = description_tokenizer.texts_to_sequences([processed_description])
            description_data_padded = pad_sequences(description_sequence, maxlen=1500)

            common_nouns = set(extract_nouns(processed_resume))
            common_nouns_str = ' '.join(common_nouns)

            common_nouns_sequence = common_nouns_tokenizer.texts_to_sequences([common_nouns_str])
            common_nouns_data = pad_sequences(common_nouns_sequence, maxlen=10)

            prediction = model.predict([resume_data_padded, description_data_padded, common_nouns_data])
            st.success(f"Your predicted ATS Score is: {prediction[0][0]:.2f}")

        except FileNotFoundError as fnf_error:
            st.error(str(fnf_error))
        except ValueError as val_error:
            st.error(str(val_error))
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please paste both your resume and job description before analyzing.")
