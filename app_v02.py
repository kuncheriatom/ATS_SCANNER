import streamlit as st
import tensorflow as tf
from transformers.file_utils import is_tf_available
from transformers import TFDistilBertModel, DistilBertTokenizer
import nltk
import re
from transformers import AutoTokenizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords



# Register your custom model class
@tf.keras.utils.register_keras_serializable()
class ATSScorePredictionModel(tf.keras.Model):
    def __init__(self):
        super(ATSScorePredictionModel, self).__init__()
        distilbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        self.distilbert_resume = distilbert_model
        self.distilbert_jd = distilbert_model
        self.dense_resume = tf.keras.layers.Dense(128, activation='relu')
        self.dense_jd = tf.keras.layers.Dense(128, activation='relu')
        self.concat_layer = tf.keras.layers.Concatenate()
        self.final_dense = tf.keras.layers.Dense(1, activation='linear')  # Regression output

    def call(self, inputs):
        resume_input, jd_input = inputs
        resume_input = resume_input[:, 0, :]  # Remove unnecessary dimensions
        jd_input = jd_input[:, 0, :]  # Remove unnecessary dimensions
        
        resume_embeddings = self.distilbert_resume(resume_input)[0]
        resume_pooled = tf.reduce_mean(resume_embeddings, axis=1)
        resume_output = self.dense_resume(resume_pooled)
        
        jd_embeddings = self.distilbert_jd(jd_input)[0]
        jd_pooled = tf.reduce_mean(jd_embeddings, axis=1)
        jd_output = self.dense_jd(jd_pooled)
        
        combined_output = self.concat_layer([resume_output, jd_output])
        final_output = self.final_dense(combined_output)
        
        return final_output

# Load your saved model
model = tf.keras.models.load_model(r'R:\Big Data Analytics  Lambton\Sem 3\AML 2034 - Bhavik Gandhi\Project\bert_model\ats_score_prediction_model1.keras')

# Load DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)  # Remove URLs, usernames, and mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)  # Join the tokens back into a single string

# Streamlit UI
st.title('ATS Score Prediction App')

# Text input for Resume and Job Description
resume_text = st.text_area("Enter Resume Text", height=200)
jd_text = st.text_area("Enter Job Description Text", height=200)

if st.button('Predict ATS Score'):
    if resume_text and jd_text:
        # Preprocess the input texts
        cleaned_resume = preprocess_text(resume_text)
        cleaned_jd = preprocess_text(jd_text)
        
        # Tokenize the inputs
        resume_inputs = tokenizer([cleaned_resume], return_tensors="tf", padding=True, truncation=True, max_length=512)
        jd_inputs = tokenizer([cleaned_jd], return_tensors="tf", padding=True, truncation=True, max_length=512)
        
        # Predict using the loaded model
        prediction = model((resume_inputs['input_ids'], jd_inputs['input_ids']))
        
        # Display the predicted score
        st.write(f"Predicted ATS Score: {prediction.numpy()[0][0]:.2f}")
    else:
        st.warning("Please enter both resume and job description text.")

