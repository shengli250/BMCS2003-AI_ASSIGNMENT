import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Configuration Parameters ---
MAX_SEQUENCE_LENGTH = 20  # Max number of words used during training
CONFIDENCE_THRESHOLD = 0.75 # New: Threshold to classify as "unrecognized intent"

# --- A. CHATBOT RESPONSE LOOKUP TABLE ---
# This dictionary maps the predicted intent name (string) to a fixed response.
RESPONSE_DICT = {
    "ask_room_price": "Our standard room price is 150 MYR per night, and a deluxe room is 250 MYR. Which room type would you like to inquire about?",
    "ask_availability": "Could you please provide the check-in and check-out dates? I can check real-time room availability for you.",
    "ask_facilities": "We offer free Wi-Fi, complimentary parking, an indoor swimming pool, and a 24-hour gym.",
    "ask_location": "Our hotel is situated in the city center, close to the central station and major shopping areas. You can find the full address on our website.",
    "ask_checkin_time": "Our standard check-in time is 3:00 PM. Please contact the front desk if you require early check-in.",
    "ask_checkout_time": "Please ensure you check out before 12:00 PM (noon). Late check-outs may incur an additional charge.",
    "ask_booking": "You can make a reservation through our official website, by calling our booking hotline, or via major online travel platforms.",
    "ask_cancellation": "Our cancellation policy depends on your booking type. Generally, cancellation is free if done 24 hours in advance.",
    "greeting": "Hello! I am happy to assist you. How may I help you with your booking or answer your questions?",
    "goodbye": "Thank you for reaching out! Have a wonderful day. Feel free to contact me if you have any other questions.",
    # Default response for unrecognized intents
    "unrecognized_intent": "I apologize, but I currently cannot understand your request. Could you please try rephrasing your question?", 
}

# --- B. Model Loading and Caching ---
# Use Streamlit's caching mechanism to load resources only once
@st.cache_resource
def load_resources():
    """Loads the model, tokenizer, and label encoder from files."""
    try:
        # Load Keras Model
        model = tf.keras.models.load_model('lstm_intent_model.h5')
        
        # Load Tokenizer
        tokenizer = joblib.load('tokenizerLSTM.joblib')
        
        # Load LabelEncoder
        le = joblib.load('label_encoder.joblib')
        
        return model, tokenizer, le
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Please ensure all files (lstm_intent_model.h5, tokenizer.joblibLSTM, label_encoder.joblib) are in the same directory. Missing file: {e.filename}")
        return None, None, None

model, tokenizer, le = load_resources()

# --- C. Prediction Function ---
def predict_intent(text):
    """
    Predicts the intent of a given text and returns the response.
    Includes logic for 'unrecognized intent' based on confidence.
    """
    if model is None or tokenizer is None or le is None:
        return "Model not loaded. Check the file paths."

    # 1. Preprocess the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, 
                                    maxlen=MAX_SEQUENCE_LENGTH, 
                                    padding='post', 
                                    truncating='post')

    # 2. Make Prediction
    predictions = model.predict(padded_sequence, verbose=0)
    
    # Get the index (ID) of the highest probability
    predicted_id = np.argmax(predictions, axis=1)[0]
    # Get the confidence score (the highest probability)
    confidence_score = np.max(predictions, axis=1)[0]
    
    # 3. Apply Confidence Threshold Logic
    if confidence_score < CONFIDENCE_THRESHOLD:
        # Intent is considered 'unrecognized'
        intent_name = "unrecognized_intent"
        # The corresponding response for 'unrecognized_intent' is retrieved from RESPONSE_DICT
        response = RESPONSE_DICT.get(intent_name)
    else:
        # Convert the predicted ID back to the intent name
        intent_name = le.inverse_transform([predicted_id])[0]
        # Retrieve the specific response for the predicted intent
        response = RESPONSE_DICT.get(intent_name, RESPONSE_DICT['unrecognized_intent'])

    # Format the confidence score to percentage
    confidence_display = f"{confidence_score*100:.2f}%"
    
    return intent_name, response, confidence_display


# --- D. Streamlit App Layout ---
def main():
    st.set_page_config(page_title="Hotel Intent Chatbot", layout="centered")

    st.title("🛎️ Hotel Intent Recognition Chatbot")
    st.markdown("Enter a user query to predict the intent and get a response.")

    st.info(f"**Confidence Threshold for Unrecognized Intent:** {CONFIDENCE_THRESHOLD*100:.0f}%")
    
    # User Input
    user_input = st.text_input("**Your Query:**", placeholder="E.g., What time is check-in?")
    
    # ***FIX: Create the button ONLY ONCE and store its state***
    button_clicked = st.button("**Get Chatbot Response**") 

    if button_clicked:
        if user_input:
            with st.spinner('Analyzing query...'):
                # Predict the intent
                intent_name, response, confidence_display = predict_intent(user_input)
                
                # --- Display Results ---
                st.markdown("---")
                
                st.subheader("💡 Analysis Result")
                
                # Highlight the predicted intent
                if intent_name == "unrecognized_intent":
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
