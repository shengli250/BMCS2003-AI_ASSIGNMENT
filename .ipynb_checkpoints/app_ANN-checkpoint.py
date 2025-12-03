import streamlit as st
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
# 导入 MLPClassifier 来获取类型提示 (可选)
from sklearn.neural_network import MLPClassifier 

# --- Configuration Parameters ---
MAX_SEQUENCE_LENGTH = 20 # Although less relevant for TFIDF, keeping it for context
CONFIDENCE_THRESHOLD = 0.70 # 设置置信度阈值，低于此值则视为“无法识别的意图”

# --- A. CHATBOT RESPONSE LOOKUP TABLE ---
RESPONSE_DICT = {
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free Wi-Fi, breakfast, pool, gym and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre (KLCC).",
    "ask_checkin_time" : "Check-in time is from 2:00 PM.",
    "ask_checkout_time" : "Check-out time is at 12:00 PM.",
    "ask_booking" : "You can book directly through our website or at the front desk.",
    "ask_cancellation" : "Cancellations are free up to 24 hours before arrival.",
    "greeting" : "Hello! How may I assist you today?",
    "goodbye" : "Goodbye! Have a great day!",
    # Default response for unrecognized intents
    "unrecognized_intent": "I apologize, but I currently cannot understand your request. Could you please try rephrasing your question?", 
}

# --- B. NLTK Download and Preprocessing Setup ---
# 使用 st.cache_resource 来确保 NLTK 资源只下载一次
@st.cache_resource(show_spinner="Downloading NLTK resources...")
def download_nltk_resources():
    """Downloads necessary NLTK resources into the Streamlit cache."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True) # This may not be necessary
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        return False

# 执行 NLTK 资源下载
if download_nltk_resources():
    # 只有下载成功后才初始化 NLTK 对象
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
else:
    # 如果下载失败，使用空集和 None 来避免后续错误
    stop_words = set()
    lemmatizer = None

def preprocess_text(text):
    """Applies the same preprocessing steps as the training script."""
    if not lemmatizer:
        return "" # Handle case where NLTK setup failed
        
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- C. Model Loading and Caching ---
@st.cache_resource
def load_resources():
    """Loads the model, vectorizer, and label encoder from files."""
    try:
        # Load MLPClassifier Model
        ann_model = joblib.load('ann_intent_model.joblib')
        
        # Load TFIDF Vectorizer
        vectorizer = joblib.load('tfidf_vectorizerANN.joblib')
        
        # Load LabelEncoder
        le = joblib.load('label_encoder.joblib')
        
        return ann_model, vectorizer, le
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Please ensure all files (ann_intent_model.joblib, tfidf_vectorizerANN.joblib, label_encoder.joblib) are in the same directory. Missing file: {e.filename}")
        return None, None, None

ann_model, vectorizer, le = load_resources()

# --- D. Prediction Function ---
def predict_intent(text):
    """
    Predicts the intent using the ANN model and applies a confidence threshold.
    """
    if ann_model is None or vectorizer is None or le is None or not lemmatizer:
        return "setup_error", RESPONSE_DICT.get("unrecognized_intent"), "N/A"

    # 1. Preprocessing and Feature Extraction
    user_input_cleaned = preprocess_text(text)
    vector = vectorizer.transform([user_input_cleaned])

    # 2. Get Probability Predictions
    # MLPClassifier provides probabilities via predict_proba
    predictions_proba = ann_model.predict_proba(vector)[0]
    
    # Get the index (ID) of the highest probability
    predicted_id = np.argmax(predictions_proba)
    # Get the confidence score (the highest probability)
    confidence_score = np.max(predictions_proba)
    
    # 3. Apply Confidence Threshold Logic
    if confidence_score < CONFIDENCE_THRESHOLD:
        intent_name = "unrecognized_intent"
        response = RESPONSE_DICT.get(intent_name)
    else:
        # Convert the predicted ID back to the intent name
        intent_name = le.inverse_transform([predicted_id])[0]
        # Retrieve the specific response for the predicted intent
        response = RESPONSE_DICT.get(intent_name, RESPONSE_DICT['unrecognized_intent'])

    confidence_display = f"{confidence_score*100:.2f}%"
    
    return intent_name, response, confidence_display


# --- E. Streamlit App Layout ---
def main():
    st.set_page_config(page_title="ANN Intent Chatbot (TF-IDF)", layout="centered")

    st.title("🤖 Hotel Intent Recognition Chatbot (ANN)")
    st.markdown("基于 **TF-IDF** 和 **Scikit-learn MLPClassifier (ANN)** 模型")

    st.info(f"**Confidence Threshold for Unrecognized Intent:** {CONFIDENCE_THRESHOLD*100:.0f}%")
    
    # User Input
    user_input = st.text_input("**Your Query:**", placeholder="E.g., I want to book a room. Do you have a gym?")
    
    # Create the button ONLY ONCE
    button_clicked = st.button("🚀 **Get Chatbot Response**") 

    if button_clicked:
        if user_input:
            with st.spinner('Analyzing query...'):
                # Predict the intent
                intent_name, response, confidence_display = predict_intent(user_input)
                
                # --- Display Results ---
                st.markdown("---")
                
                st.subheader("💡 Analysis Result")
                
                # Highlight the predicted intent
                if intent_name == "unrecognized_intent" or intent_name == "setup_error":
                    st.error(f"**Predicted Intent:** `{intent_name}` (Confidence: {confidence_display})")
                else:
                    st.success(f"**Predicted Intent:** `{intent_name}` (Confidence: {confidence_display})")

                st.subheader("💬 Chatbot Response")
                st.markdown(f"> **{response}**")
                
        else:
            # Handle the case where the button is clicked but the input is empty
            st.warning("Please enter a query to get a response.")

if __name__ == "__main__":
    main()
