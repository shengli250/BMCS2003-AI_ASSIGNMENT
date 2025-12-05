import streamlit as st
import numpy as np
import joblib
import nltk
import random
import json
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
# Optional: Import MLPClassifier for type hinting
from sklearn.neural_network import MLPClassifier 

# --- Configuration Parameters ---
MAX_SEQUENCE_LENGTH = 20 # Although less relevant for TFIDF, keeping it for context
CONFIDENCE_THRESHOLD = 0.50 # Lowered slightly to be more forgiving, adjust as needed

# --- A. LOAD RESPONSES FROM JSON ---
@st.cache_data
def load_response_json():
    """Loads the response configuration from the JSON file."""
    try:
        with open('response.json', 'r', encoding='utf-8') as f:
            responses = json.load(f)
        # Add a default fallback if not present in file
        if "unrecognized_intent" not in responses:
            responses["unrecognized_intent"] = "I apologize, but I currently cannot understand your request. Could you please try rephrasing your question?"
        return responses
    except FileNotFoundError:
        st.error("response.json file not found. Please upload it.")
        return {"unrecognized_intent": "System Error: Responses not loaded."}
    except json.JSONDecodeError:
        st.error("Error decoding response.json. Please check the file format.")
        return {"unrecognized_intent": "System Error: Invalid JSON format."}

RESPONSE_DICT = load_response_json()

# --- PROMPT MAPPING ---
# Mapping of intent keys to user-friendly natural language prompts for buttons
# This list now covers all intents found in the complete dataset/json
PROMPT_MAPPING = {
    # General Info
    "ask_room_price": "What are the room rates?",
    "ask_availability": "Do you have rooms available?",
    "ask_facilities": "What facilities do you have?",
    "ask_location": "Where is the hotel located?",
    
    # Check-in/out & Booking
    "ask_checkin_time": "What time is check-in?",
    "ask_checkout_time": "What time is check-out?",
    "ask_booking": "How can I book a room?",
    "ask_cancellation": "What is the cancellation policy?",
    
    # Policies & Services
    "ask_pet_policy": "Are pets allowed?",
    "ask_smoking_policy": "Is smoking allowed?",
    "ask_child_policy": "What is the policy for children?",
    "ask_luggage_storage": "Can I store my luggage?",
    "ask_breakfast_details": "Is breakfast included?",
    "ask_airport_transfer": "Do you offer airport pickup?",
    
    # Assistance
    "ask_nearby_attractions": "What is nearby?",
    "ask_lost_item": "I lost an item, can you help?",
    
    # Social (Usually not used for buttons, but mapped for completeness)
    "greeting": "Hello!",
    "goodbye": "Goodbye!"
}

# Valid intents to be used for random suggestions
# We exclude 'greeting' and 'goodbye' from the random suggestions buttons as they are conversational
EXCLUDED_FROM_SUGGESTIONS = ["greeting", "goodbye"]

# Filter keys to ensure they actually exist in the loaded RESPONSE_DICT or PROMPT_MAPPING
SUGGESTED_INTENTS = [
    key for key in PROMPT_MAPPING.keys() 
    if key not in EXCLUDED_FROM_SUGGESTIONS and (key in RESPONSE_DICT or key in PROMPT_MAPPING)
]

# --- B. NLTK Download and Preprocessing Setup ---
# Use st.cache_resource to ensure NLTK resources are downloaded only once
@st.cache_resource(show_spinner="Downloading NLTK resources...")
def download_nltk_resources():
    """Downloads necessary NLTK resources into the Streamlit cache."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        return False

# Execute NLTK resource download
if download_nltk_resources():
    # Only initialize NLTK objects if download was successful
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
else:
    # If download fails, use empty set and None to avoid subsequent errors
    stop_words = set()
    lemmatizer = None

def preprocess_text(text):
    """Applies the same preprocessing steps as the training script."""
    if not lemmatizer:
        return "" # Handle case where NLTK setup failed
        
    text = str(text).lower()
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
        # Load the model trained with the dense matrix
        ann_model = joblib.load('ann_intent_model_dense.joblib')
        
        # Ensure the correct TFIDF Vectorizer filename is loaded
        vectorizer = joblib.load('ann_tfidf_vectorizer_dense.joblib')
        
        # Ensure the correct LabelEncoder filename is loaded
        le = joblib.load('ann_label_encoder_dense.joblib')
        
        return ann_model, vectorizer, le
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Please ensure all files (ann_intent_model_dense.joblib, ann_tfidf_vectorizer_dense.joblib, ann_label_encoder_dense.joblib) are in the same directory. Missing file: {e.filename}")
        return None, None, None

ann_model, vectorizer, le = load_resources()

# --- D. Prediction Function (With Response Time) ---
def predict_intent(text):
    """
    Predicts the intent using the ANN model and applies a confidence threshold.
    Also measures the time taken for the prediction.
    """
    start_time = time.time() # Start timer

    if ann_model is None or vectorizer is None or le is None or not lemmatizer:
        end_time = time.time()
        return "setup_error", RESPONSE_DICT.get("unrecognized_intent"), "N/A", end_time - start_time

    # 1. Preprocessing and Feature Extraction (Sparse Matrix)
    user_input_cleaned = preprocess_text(text)
    vector = vectorizer.transform([user_input_cleaned])

    # Convert sparse TFIDF vector to dense matrix, as MLPClassifier was trained on dense data
    vector_dense = vector.toarray()
    
    # 2. Get Probability Predictions
    # MLPClassifier provides probabilities via predict_proba
    predictions_proba = ann_model.predict_proba(vector_dense)[0] # Use dense matrix
    
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
        response = RESPONSE_DICT.get(intent_name, f"I understood the intent is '{intent_name}', but I don't have a response for it in the database.")

    confidence_display = f"{confidence_score*100:.2f}%"
    
    end_time = time.time() # End timer
    response_time = end_time - start_time # Calculate duration
    
    return intent_name, response, confidence_display, response_time


# --- E. Streamlit App Layout (with Chat History) ---
def main():
    st.set_page_config(page_title="Hotel AI Assistant", layout="centered")

    st.title("🤖 Grand Hotel FAQ Chatbot (MLPClassifier Model)")
    st.markdown("Ask me about room rates, availability, facilities, and more!")

    # 1. Initialize chat history (Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add an initial greeting message
        # Check if 'greeting' exists in loaded dictionary, otherwise use default
        greeting_text = RESPONSE_DICT.get('greeting', "Hello! How may I assist you today?")
        st.session_state.messages.append({"role": "assistant", "content": greeting_text})

    # Initialize state for handling button clicks (Suggested Questions)
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # 2. Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "intent" in message:
                # Display intent, confidence, AND response time
                st.caption(f"Intent: **{message['intent']}** | Confidence: **{message['confidence']}** | Time: **{message['time']:.4f}s**")
            st.markdown(message["content"])

    # --- 3. Suggested Questions (Buttons) ---
    # Select 4 random intents for suggestions (increased to 4 for better variety)
    if SUGGESTED_INTENTS:
        # Ensure we don't try to select more than available
        num_suggestions = min(4, len(SUGGESTED_INTENTS))
        random_intents = random.sample(SUGGESTED_INTENTS, num_suggestions)

        st.markdown("**Suggested Questions:**")
        cols = st.columns(num_suggestions)

        # Iterate through random intents to create buttons
        for i, intent_key in enumerate(random_intents):
            prompt_text = PROMPT_MAPPING.get(intent_key, intent_key)
            with cols[i]:
                # Check if the button is clicked, and if so, set the input text
                if st.button(prompt_text, key=f"btn_{intent_key}", use_container_width=True):
                    # Store the button text in session_state and trigger a rerun
                    st.session_state.pending_input = prompt_text
                    st.rerun()

    # --- 4. Handle User/Button Input ---
    user_input = None

    # Priority check: If there is a pending input from a button click
    if st.session_state.pending_input:
        user_input = st.session_state.pending_input
        # Clear the pending input immediately after retrieval
        st.session_state.pending_input = None
    else:
        user_input = st.chat_input("How can I help you?")

    if user_input:
        # 4a. Add user input to history and display
        st.session_state.messages.append({"role": "user", "content": user_input})

    # If the history was just updated, process the *last* user message
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        current_user_input = st.session_state.messages[-1]["content"]

        # 4b. Perform prediction and generate reply
        with st.spinner('Analyzing query...'):
            intent_name, response, confidence_display, response_time = predict_intent(current_user_input)
            
            # 4c. Add assistant reply to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "intent": intent_name,
                "confidence": confidence_display,
                "time": response_time
            })
            
            # Rerun again to show the assistant's response in the main loop display
            st.rerun()

            # 4d. Display assistant reply on the interface
            with st.chat_message("assistant"):
                # Highlight intent, confidence, and time
                st.caption(f"Intent: **{intent_name}** | Confidence: **{confidence_display}** | Time: **{response_time:.4f}s**")
                st.markdown(response)

if __name__ == "__main__":
    main()