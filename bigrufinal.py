import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, GRU, Dense, Dropout, concatenate, SpatialDropout1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Load dataset
data = pd.read_csv('dbhealthcareats.csv')

# Handle missing values
data.dropna(inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
data['Category'] = encoder.fit_transform(data['Category'])

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

# Abbreviations dictionary for job market
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

# Function to expand abbreviations
def expand_abbreviations(text, abbreviations):
    for abbr, expanded in abbreviations.items():
        text = re.sub(r'\b{}\b'.format(abbr), expanded, text)
    return text

# Function to clean and preprocess text
def clean_and_preprocess(text):
    text = expand_abbreviations(text, abbreviations)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(tokens)

# Function to extract nouns
def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.lemma_ for token in doc if token.pos_ == "NOUN"]
    return nouns

# Apply cleaning and preprocessing
data['processed_resume'] = data['Resume_str'].apply(clean_and_preprocess)
data['processed_description'] = data['description'].apply(clean_and_preprocess)

max_sequence_length = 1500

# Extract nouns separately for resumes and job descriptions
data['resume_nouns'] = data['processed_resume'].apply(extract_nouns)
data['description_nouns'] = data['processed_description'].apply(extract_nouns)

# Find common nouns between resume and job description
data['common_nouns'] = data.apply(lambda row: list(set(row['resume_nouns']).intersection(set(row['description_nouns']))), axis=1)

# Convert list of common nouns to string
data['common_nouns_str'] = data['common_nouns'].apply(lambda x: ' '.join(x))

# Tokenizer and sequences for common nouns
common_nouns_tokenizer = Tokenizer()
common_nouns_tokenizer.fit_on_texts(data['common_nouns_str'])
common_nouns_sequences = common_nouns_tokenizer.texts_to_sequences(data['common_nouns_str'])
max_common_nouns_length = 10  # Adjust based on your data
common_nouns_data = pad_sequences(common_nouns_sequences, maxlen=max_common_nouns_length)

# Prepare other data
resume_tokenizer = Tokenizer()
resume_tokenizer.fit_on_texts(data['processed_resume'])
resume_sequences = resume_tokenizer.texts_to_sequences(data['processed_resume'])
resume_data_padded = pad_sequences(resume_sequences, maxlen=max_sequence_length)

description_tokenizer = Tokenizer()
description_tokenizer.fit_on_texts(data['processed_description'])
description_sequences = description_tokenizer.texts_to_sequences(data['processed_description'])
description_data_padded = pad_sequences(description_sequences, maxlen=max_sequence_length)

# Target variable
y = data['ATS_Score'].values

# Save the tokenizers to a pickle file
tokenizers = {
    'resume_tokenizer': resume_tokenizer,
    'description_tokenizer': description_tokenizer,
    'common_nouns_tokenizer': common_nouns_tokenizer
}
save_path = r'tokenizershealth.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(tokenizers, f)

# Train-test split
X_train_resume, X_test_resume, X_train_description, X_test_description, X_train_common, X_test_common, y_train, y_test = train_test_split(
    resume_data_padded, description_data_padded, common_nouns_data, y, test_size=0.2, random_state=42
)

# Further split for validation
X_train_resume, X_val_resume, X_train_description, X_val_description, X_train_common, X_val_common, y_train, y_val = train_test_split(
    X_train_resume, X_train_description, X_train_common, y_train, test_size=0.2, random_state=42
)

# Model Inputs
resume_input = Input(shape=(max_sequence_length,), name='resume_input')
description_input = Input(shape=(max_sequence_length,), name='description_input')
common_input = Input(shape=(max_common_nouns_length,), name='common_input')

# Embedding Layer for resume
embedding_dim = 100  # Adjust based on your embedding size
resume_embedding_layer = Embedding(input_dim=len(resume_tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length)(resume_input)
description_embedding_layer = Embedding(input_dim=len(description_tokenizer.word_index) + 1, output_dim=embedding_dim, input_length=max_sequence_length)(description_input)

# Apply Spatial Dropout
resume_embedded_dropout = SpatialDropout1D(0.2)(resume_embedding_layer)
description_embedded_dropout = SpatialDropout1D(0.2)(description_embedding_layer)

# Bidirectional GRU Layers
resume_gru = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(resume_embedded_dropout)
description_gru = Bidirectional(GRU(128, return_sequences=True, dropout=0.3))(description_embedded_dropout)

# Global Average Pooling
resume_gru_pool = GlobalAveragePooling1D()(resume_gru)
description_gru_pool = GlobalAveragePooling1D()(description_gru)

# Dense Layers for Common Input
common_dense = Dense(32, activation='relu')(common_input)

# Combine GRU outputs with common dense layer
combined = concatenate([resume_gru_pool, description_gru_pool, common_dense])

# Additional Dense Layers with Regularization
x = Dense(64, activation='relu')(combined)
x = Dropout(0.3)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)

# Output Layer
output = Dense(1, activation='linear')(x)

# Model definition
model = Model(inputs=[resume_input, description_input, common_input], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Training the modified model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

history = model.fit(
    [X_train_resume, X_train_description, X_train_common], y_train,
    epochs=5,  # Adjust epochs as needed
    batch_size=64,  # Adjust batch size as needed
    validation_data=([X_val_resume, X_val_description, X_val_common], y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint]
)

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate([X_test_resume, X_test_description, X_test_common], y_test)
print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

# Save the model
model_save_path = r'bighealth2.keras'
model.save(model_save_path)
print(f"Model saved successfully to {model_save_path}.")

# Predict on the test set
y_pred = model.predict([X_test_resume, X_test_description, X_test_common])
