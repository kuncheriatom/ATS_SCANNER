

# ATS Score Generation Project

## Project Overview

The ATS Score Generation project aims to revolutionize the recruitment process by creating an AI-powered tool that evaluates resumes against job descriptions. By leveraging advanced Natural Language Processing (NLP) techniques and machine learning models, the tool provides accurate, unbiased, and consistent assessments, helping match talent with suitable job opportunities.

## Key Features

- **Bidirectional GRU and LSTM Models**: These models are used to capture the nuances in resume and job description matching, improving the accuracy of the ATS score prediction.
- **Text Preprocessing**: Includes text cleaning, abbreviation expansion, tokenization, and sequence padding to prepare the data for model input.
- **Feature Engineering**: Focuses on extracting common nouns between resumes and job descriptions as additional features to enhance model performance.
- **Model Interpretability**: Utilizes LIME (Local Interpretable Model-agnostic Explanations) to explain individual predictions, ensuring transparency and fairness.

## Methodology

1. **Text Preprocessing**: 
   - Cleaning and expanding abbreviations.
   - Tokenizing and padding sequences.
   - Extracting common nouns as additional features.

2. **Model Development**:
   - **Bidirectional GRU Model**: Processes input text in both directions, capturing context from past and future words.
   - **Bidirectional LSTM Model**: Similar to GRU but with LSTM layers, providing a different approach to capturing long-term dependencies in the text.
   - Both models are fine-tuned to optimize Mean Squared Error (MSE) and Mean Absolute Error (MAE).

3. **Model Tuning**:
   - Experimented with different model versions, adjusting input layers, embedding dimensions, GRU/LSTM units, dropout rates, and pooling strategies.

4. **Model Deployment**:
   - The final model is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/mednow/ATS) for user access.

## Installation

### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- SpaCy

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ats-score-generator.git
   cd ats-score-generator
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the model:
     ```bash
   streamlit run app.py
   ```


## Usage

- **Score Generation**: Input a resume and a job description, and the tool will generate an ATS score indicating the match between them.
