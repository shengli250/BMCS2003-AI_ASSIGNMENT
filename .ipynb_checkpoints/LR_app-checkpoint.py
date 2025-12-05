import streamlit as st
import numpy as np
import joblib
import nltk
import random
import json
import time
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.50  # Threshold for accepting a prediction

# --- 1. Load Responses from JSON ---
@st.cache_data
def load_response_json():
    """Loads the response configuration from the JSON file."""
    try:
        with open('response.json', 'r', encoding='utf-8') as f:
            responses = json.load(f)
        if "unrecognized_intent" not in responses:
            responses["unrecognized_intent"] = "I'm sorry, I didn't quite catch that. Could you please rephrase?"
        return responses
    except FileNotFoundError:
        st.error("response.json file not found.")
        return {}

RESPONSE_DICT = load_response_json()

# --- 2. Dynamic Prompt Mapping ---
# Create user-friendly prompts from intents found in the JSON
PROMPT_MAPPING = {
    "ask_room_price": "What are the room rates?",
    "ask_availability": "Do you have rooms available?",
    "ask_facilities": "What facilities do you have?",
    "ask_location": "Where is the hotel located?",
    "ask_checkin_time": "What time is check-in?",
    "ask_checkout_time": "What time is check-out?",
    "ask_booking": "How can I book a room?",
    "ask_cancellation": "Cancellation policy?",
    "ask_pet_policy": "Are pets allowed?",
    "ask_breakfast_details": "Is breakfast included?",
    "ask_wifi": "Is there free Wi-Fi?",
    "ask_parking": "Do you have parking?",
    "greeting": "Hello!",
    "goodbye": "Goodbye!"
}

# Generate a list of valid intents for suggestions (excluding simple greetings)
SUGGESTED_INTENTS = [
    key for key in PROMPT_MAPPING.keys() 
    if key in RESPONSE_DICT and key not in ["greeting", "goodbye"]
]

# --- 3. NLTK Setup ---
@st.cache_resource
def setup_nltk():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"NLTK Download Error: {e}")
        return False

if setup_nltk():
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
else:
    stop_words = set()
    lemmatizer = None

def preprocess_text(text):
    """
    Preprocessing function. 
    MUST match the logic used in 'train_chatbot_lr.py'.
    """
    if not lemmatizer: return text
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- 4. Load LR Model & Vectorizer ---
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load('logistic_regression_intent_model.joblib')
        vectorizer = joblib.load('tfidf_vectorizerLR.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Please run 'train_chatbot_lr.py' first.")
        return None, None

lr_model, vectorizer = load_model_resources()

# --- 5. Prediction Function ---
def predict_intent(text):
    start_time = time.time()
    
    if not lr_model or not vectorizer:
        return "error", "System Error: Model not loaded.", "0%", 0

    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Vectorize
    vector = vectorizer.transform([cleaned_text])
    
    # Predict Probabilities (Confidence)
    # lr_model.predict_proba returns an array of shape (1, n_classes)
    probs = lr_model.predict_proba(vector)[0]
    
    # Get the index of the max probability
    max_index = np.argmax(probs)
    confidence = probs[max_index]
    
    # Get the class label (Intent Name)
    predicted_intent = lr_model.classes_[max_index]
    
    # Calculate Response Time
    end_time = time.time()
    response_time = end_time - start_time

    # Threshold Check
    if confidence < CONFIDENCE_THRESHOLD:
        return "unrecognized_intent", RESPONSE_DICT.get("unrecognized_intent"), f"{confidence*100:.2f}%", response_time
    
    # Get Response
    response_text = RESPONSE_DICT.get(predicted_intent, "I understood what you said, but I don't have a response prepared.")
    
    return predicted_intent, response_text, f"{confidence*100:.2f}%", response_time

# --- 6. Streamlit UI ---
def main():
    st.set_page_config(page_title="Logistic Regression Chatbot", layout="centered")
    
    st.title("🤖 Hotel Assistant (Logistic Regression)")
    st.markdown("Fast and efficient intent classification.")

    # Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        greeting = RESPONSE_DICT.get("greeting", "Hello! How can I help?")
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and "intent" in msg:
                st.caption(f"Intent: **{msg['intent']}** | Conf: **{msg['confidence']}** | Time: **{msg['time']:.4f}s**")
            st.markdown(msg["content"])

    # Suggested Questions
    if SUGGESTED_INTENTS:
        # Show 3 random suggestions
        suggestions = random.sample(SUGGESTED_INTENTS, min(3, len(SUGGESTED_INTENTS)))
        st.markdown("---")
        st.markdown("**Try asking:**")
        cols = st.columns(len(suggestions))
        for i, intent in enumerate(suggestions):
            label = PROMPT_MAPPING.get(intent, intent)
            with cols[i]:
                if st.button(label, key=f"btn_{intent}", use_container_width=True):
                    st.session_state.pending_input = label
                    st.rerun()

    # Input Handling
    user_input = None
    if st.session_state.pending_input:
        user_input = st.session_state.pending_input
        st.session_state.pending_input = None
    else:
        user_input = st.chat_input("Type your message here...")

    # Process Input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            intent, response, conf, resp_time = predict_intent(user_input)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "intent": intent,
                "confidence": conf,
                "time": resp_time
            })
            st.rerun()

if __name__ == "__main__":
    main()