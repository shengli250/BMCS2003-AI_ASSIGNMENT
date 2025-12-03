import streamlit as st
import pandas as pd
import numpy as np
import nltk
from joblib import load
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# --- NLTK Resource Setup (保持不变) ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/stopwords')
except LookupError:
    st.info("Downloading NLTK resources...")
    try:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')
        st.success("NLTK Resources downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        st.stop()

# Initialize NLTK resources
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# --- Preprocessing Function (保持不变) ---
def preprocess_text(text):
    if not isinstance(text, str):
        return "" 
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Chatbot Core Function (重点修改这里) ---
# Predefined fixed responses (Retrieval System)
RESPONSES = {
    "ask_room_price": "Our rooms start from RM180 per night.",
    "ask_availability": "We currently have several rooms available.",
    "ask_facilities": "We offer free Wi-Fi, breakfast, pool, gym and parking.",
    "ask_location": "We are located in Kuala Lumpur City Centre (KLCC).",
    "ask_checkin_time" : "Check-in time is from 2:00 PM.",
    "ask_checkout_time" : "Check-out time is at 12:00 PM.",
    "ask_booking" : "You can book directly through our website or at the front desk.",
    "ask_cancellation" : "Cancellations are free up to 24 hours before arrival.",
    "greeting" : "Hello! How may I assist you today?",
    "goodbye" : "Goodbye! Have a great day!"
}

# 🌟 关键修改：添加置信度检查
def chatbot_reply_nb(user_input, model, vectorizer, responses):
    # 1. Preprocessing
    cleaned_input = preprocess_text(user_input)

    # 2. Feature Extraction
    if not cleaned_input:
        return "Please provide a valid question.", "Empty Input", 1.0 # 返回一个明确的错误信息

    vector = vectorizer.transform([cleaned_input])

    # 3. Intent Prediction and Confidence
    # Naive Bayes 模型的 predict_proba 返回每个类别的概率
    probabilities = model.predict_proba(vector)[0]
    intent_index = np.argmax(probabilities)
    confidence = probabilities[intent_index]
    intent = model.classes_[intent_index]

    # 🌟 设置置信度阈值 (可调整)
    CONFIDENCE_THRESHOLD = 0.3 
    
    # 4. Retrieval and Fallback Logic
    if confidence < CONFIDENCE_THRESHOLD:
        # 🌟 低置信度回退
        response = "I'm sorry, I don't seem to understand that question. Could you please rephrase or ask about price, availability, or facilities?"
        predicted_intent = "Fallback (Low Confidence)"
    else:
        # 高置信度，检查是否有预设回复
        response = responses.get(intent, 
            f"Sorry, I predicted the intent **'{intent}'** with high confidence ({confidence:.2f}), but I don't have a specific response for that yet. Please rephrase your question."
        )
        predicted_intent = intent
    
    return response, predicted_intent, confidence

# --- Streamlit Application Layout and Logic ---

# Title and Description
st.title("🛎️ Hotel Booking Intent Chatbot")
st.markdown("""
This chatbot uses a **Multinomial Naive Bayes** model trained on TF-IDF features 
to classify user intent and provide a relevant, predefined response.
**New:** It now includes a **confidence check** to provide a fallback message if it's unsure.
""")

# Load Model and Vectorizer (Use st.cache_resource for efficiency)
@st.cache_resource
def load_resources():
    model_path = 'naive_bayes_intent_model.joblib'
    vectorizer_path = 'tfidf_vectorizer.joblib'
    
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        st.error(f"Required files not found: '{model_path}' and/or '{vectorizer_path}'.")
        st.warning("Please ensure you run your training script first to save the model and vectorizer.")
        st.stop()
        
    try:
        model = load(model_path)
        vectorizer = load(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

nb_model, vectorizer = load_resources()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about the hotel (e.g., 'What is the check-in time?'):"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate chatbot response
    with st.chat_message("assistant"):
        # 🌟 关键修改：接收置信度
        response, predicted_intent, confidence = chatbot_reply_nb(prompt, nb_model, vectorizer, RESPONSES)
        
        # Display the main response
        st.markdown(response)
        
        # Display the predicted intent and confidence (即使是回退也显示)
        st.caption(f"🤖 Predicted Intent: **{predicted_intent}** | Confidence: **{confidence:.2f}**")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Optional: Display a list of known intents for guidance
st.sidebar.header("Known Intents")
st.sidebar.info("The chatbot can answer questions related to these topics:")
st.sidebar.text(f"- {', '.join(RESPONSES.keys())}")
