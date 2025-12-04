import streamlit as st
import numpy as np
import joblib
import nltk
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
# Optional: Import MLPClassifier for type hinting
from sklearn.neural_network import MLPClassifier 

# --- Configuration Parameters ---
MAX_SEQUENCE_LENGTH = 20 # Although less relevant for TFIDF, keeping it for context
CONFIDENCE_THRESHOLD = 0.70 # Set confidence threshold; if prediction score is below this, intent is "unrecognized"

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

# Mapping of intent keys to user-friendly natural language prompts for buttons
PROMPT_MAPPING = {
    "ask_room_price": "What is the room price?",
    "ask_availability": "Are rooms available?",
    "ask_facilities": "Tell me about the facilities.",
    "ask_location": "What is the hotel's location?",
    "ask_checkin_time" : "What time is check-in?",
    "ask_checkout_time" : "What time is check-out?",
    "ask_booking" : "How do I book a room?",
    "ask_cancellation" : "What is the cancellation policy?",
}

# Valid intents to be used for random suggestions
SUGGESTED_INTENTS = list(PROMPT_MAPPING.keys())

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
        # Load the model trained with the dense matrix
        ann_model = joblib.load('ann_intent_model_dense.joblib')
        
        # Ensure the correct TFIDF Vectorizer filename is loaded
        vectorizer = joblib.load('ann_tfidf_vectorizer_dense.joblib')
        
        # Ensure the correct LabelEncoder filename is loaded
        le = joblib.load('ann_label_encoder_dense.joblib')
        
        return ann_model, vectorizer, le
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Please ensure all files (ann_intent_model_dense.joblib, tfidf_vectorizer_dense.joblib, label_encoder_dense.joblib) are in the same directory. Missing file: {e.filename}")
        return None, None, None

ann_model, vectorizer, le = load_resources()

# --- D. Prediction Function ---
def predict_intent(text):
    """
    Predicts the intent using the ANN model and applies a confidence threshold.
    """
    if ann_model is None or vectorizer is None or le is None or not lemmatizer:
        return "setup_error", RESPONSE_DICT.get("unrecognized_intent"), "N/A"

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
        response = RESPONSE_DICT.get(intent_name, RESPONSE_DICT['unrecognized_intent'])

    confidence_display = f"{confidence_score*100:.2f}%"
    
    return intent_name, response, confidence_display


# --- E. Streamlit App Layout (with Chat History) ---
def main():
    st.set_page_config(page_title="ANN Intent Chatbot (Chat History)", layout="centered")

    st.title("🤖 Hotel Chatbot (ANN/MLP)")
    st.caption(f"Confidence Threshold: **{CONFIDENCE_THRESHOLD*100:.0f}%**")

    # 1. Initialize chat history (Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add an initial greeting message
        st.session_state.messages.append({"role": "assistant", "content": RESPONSE_DICT['greeting']})

    # Initialize state for handling button clicks (Suggested Questions)
    if "pending_input" not in st.session_state:
        st.session_state.pending_input = None

    # 2. Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "intent" in message:
                st.caption(f"Intent: **{message['intent']}** | Confidence: **{message['confidence']}**")
            st.markdown(message["content"])

    # --- 3. Suggested Questions (Buttons) ---
    # Select 3 random intents for suggestions
    if SUGGESTED_INTENTS:
        # Ensure we don't try to select more than available
        num_suggestions = min(3, len(SUGGESTED_INTENTS))
        random_intents = random.sample(SUGGESTED_INTENTS, num_suggestions)

        st.markdown("**Suggested Questions:**")
        cols = st.columns(num_suggestions)

        # Iterate through random intents to create buttons
        for i, intent_key in enumerate(random_intents):
            prompt_text = PROMPT_MAPPING[intent_key]
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
            intent_name, response, confidence_display = predict_intent(current_user_input)
            
            # 4c. Add assistant reply to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "intent": intent_name,
                "confidence": confidence_display
            })
            
            # Rerun again to show the assistant's response in the main loop display
            st.rerun()

            # 4d. Display assistant reply on the interface
            with st.chat_message("assistant"):
                # Highlight intent and confidence
                st.caption(f"Intent: **{intent_name}** | Confidence: **{confidence_display}**")
                st.markdown(response)

if __name__ == "__main__":
    main()