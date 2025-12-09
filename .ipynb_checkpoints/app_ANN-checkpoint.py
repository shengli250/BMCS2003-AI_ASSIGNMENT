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
        with open('responses.json', 'r', encoding='utf-8') as f:
            responses = json.load(f)
        return responses
    except FileNotFoundError:
        st.error("responses.json file not found. Please upload it.")
        return {}
    except json.JSONDecodeError:
        st.error("Error decoding responses.json. Please check the file format.")
        return {}

RESPONSE_DICT = load_response_json()

# --- PROMPT MAPPING ---
# Mapping of intent keys to user-friendly natural language prompts for buttons
# This list now covers all intents found in the complete dataset/json
PROMPT_MAPPING = {
    # General
    "check_functions": "What services do you offer?",
    "human_agent": "Can I speak to a human agent?",

    # Booking & Rates
    "book_hotel": "I want to book a room.",
    "check_hotel_prices": "What are your room rates?",
    "check_room_availability": "Do you have rooms available?",
    "check_room_type": "What types of rooms do you have?",
    "check_hotel_offers": "Do you have any special offers?",
    "search_hotel": "I am looking for a hotel details.",

    # Reservation Management
    "check_hotel_reservation": "Can I check my booking details?",
    "change_hotel_reservation": "I need to modify my reservation.",
    "cancel_hotel_reservation": "I want to cancel my booking.",
    "cancellation_fees": "Is there a cancellation fee?",
    "add_night": "Can I extend my stay?",
    "get_refund": "How do I request a refund?",

    # Check-in/out & Stay
    "check_in": "What time is check-in?",
    "check_out": "What time is check-out?",
    "invoices": "Can I get a copy of my invoice?",
    "check_payment_methods": "What payment methods are accepted?",

    # Policies
    "bring_pets": "Are pets allowed?",
    "check_child_policy": "What is the policy for children?",
    "check_smoking_policy": "Is smoking allowed in rooms?",
    
    # Facilities & Services
    "check_hotel_facilities": "What facilities does the hotel have?",
    "book_parking_space": "Do you have parking available?",
    "shuttle_service": "Do you offer airport shuttle?",
    "store_luggage": "Can I store my luggage?",
    "check_menu": "Can I see the restaurant menu?",
    "host_event": "I would like to host an event.",
    
    # Support & Feedback
    "customer_service": "Contact customer service.",
    "check_lost_item": "I lost an item, can you help?",
    "file_complaint": "I want to file a complaint.",
    "leave_review": "Where can I leave a review?",
    "redeem_points": "How do I redeem loyalty points?",
    "check_nearby_attractions": "What attractions are nearby?"
}

# Valid intents to be used for random suggestions
# We exclude 'greeting' and 'goodbye' from the random suggestions buttons as they are conversational
EXCLUDED_FROM_SUGGESTIONS = ["greeting", "goodbye", "unknown_intent"]

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

def preprocess_instruction(instruction):
    """Applies the same preprocessing steps as the training script."""
    if not lemmatizer:
        return "" # Handle case where NLTK setup failed
        
    instruction = str(instruction).lower()
    instruction = re.sub(r'[^\w\s]', '', instruction)
    tokens = word_tokenize(instruction)
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
def predict_intent(instruction):
    """
    Predicts the intent using the ANN model and applies a confidence threshold.
    Also measures the time taken for the prediction.
    """
    start_time = time.time() # Start timer

    if ann_model is None or vectorizer is None or le is None or not lemmatizer:
        end_time = time.time()
        return "setup_error", RESPONSE_DICT.get("unknown_intent"), "N/A", end_time - start_time

    # 1. Preprocessing and Feature Extraction (Sparse Matrix)
    user_input_cleaned = preprocess_instruction(instruction)
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
        intent_name = "unknown_intent"
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

    st.title("🏨 Astra Imperium Hotel FAQ Chatbot (MLPClassifier Model)")
    st.markdown("Ask me about room rates, availability, facilities, and more!")

    # 1. Initialize chat history (Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add an initial greeting message
        # Check if 'greeting' exists in loaded dictionary, otherwise use default
        greeting_instruction = RESPONSE_DICT.get('greeting', "Hello! How may I assist you today?")
        st.session_state.messages.append({"role": "assistant", "content": greeting_instruction})

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
    if "random_intents" not in st.session_state:
        # Only generate new random intents if they are not already stored
        if SUGGESTED_INTENTS:
            num_suggestions = min(4, len(SUGGESTED_INTENTS))
            # Save the random sample to session state
            st.session_state.random_intents = random.sample(SUGGESTED_INTENTS, num_suggestions)
        else:
            st.session_state.random_intents = []

    # Retrieve the persistent list of intents from the session state
    current_intents = st.session_state.random_intents

    if current_intents:
        st.markdown("**Suggested Questions:**")
        cols = st.columns(len(current_intents))

        # Iterate through the stored intents to create buttons
        for i, intent_key in enumerate(current_intents):
            prompt_instruction = PROMPT_MAPPING.get(intent_key, intent_key)
            with cols[i]:
                # Check if the button is clicked
                if st.button(prompt_instruction, key=f"btn_{intent_key}", use_container_width=True):
                    # 1. Capture the user's selection into pending_input
                    st.session_state.pending_input = prompt_instruction
                    
                    # Clear the stored random intents.This ensures that AFTER the bot answers, a FRESH set of random suggestions will be generated for the next turn.
                    if "random_intents" in st.session_state:
                        del st.session_state.random_intents
                    
                    # Force an immediate rerun to process the pending input
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