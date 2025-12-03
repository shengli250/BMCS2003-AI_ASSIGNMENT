import streamlit as st
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.ensemble import RandomForestClassifier # 导入模型类型

# --- Configuration Parameters ---
# Random Forest 概率通常较高，我们将阈值设为 0.80 
CONFIDENCE_THRESHOLD = 0.7

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

# --- B. NLTK Download and Preprocessing Setup (Cached) ---
@st.cache_resource(show_spinner="Downloading NLTK resources...")
def download_nltk_resources():
    """Downloads necessary NLTK resources and initializes objects."""
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        return True, stop_words, lemmatizer
    except Exception as e:
        st.error(f"Failed to download NLTK resources: {e}")
        return False, set(), None

is_nltk_ready, stop_words, lemmatizer = download_nltk_resources()

def preprocess_text(text):
    """Applies the same preprocessing steps as the training script."""
    if not lemmatizer:
        return "" 
        
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- C. Model Loading and Caching ---
@st.cache_resource
def load_resources():
    """Loads the model and vectorizer from files."""
    try:
        # Load Random Forest Model
        rf_model = joblib.load('random_forest_intent_model.joblib')
        
        # Load TFIDF Vectorizer
        # 确保文件名与训练代码保存的一致
        vectorizer = joblib.load('tfidf_vectorizerRF.joblib') 
        
        return rf_model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Error loading required model files. Please ensure all files (random_forest_intent_model.joblib, tfidf_vectorizerRF.joblib) are in the same directory. Missing file: {e.filename}")
        return None, None

rf_model, vectorizer = load_resources()

# --- D. Prediction Function ---
def predict_intent(text):
    """
    Predicts the intent using the Random Forest model and applies a confidence threshold.
    """
    if rf_model is None or vectorizer is None or not is_nltk_ready:
        return "setup_error", RESPONSE_DICT.get("unrecognized_intent"), "N/A"

    # 1. Preprocessing and Feature Extraction
    user_input_cleaned = preprocess_text(text)
    vector = vectorizer.transform([user_input_cleaned])

    # 2. Get Probability Predictions
    # RandomForestClassifier provides probabilities via predict_proba
    predictions_proba = rf_model.predict_proba(vector)[0]
    
    # Get the index (ID) of the highest probability
    predicted_index = np.argmax(predictions_proba)
    # Get the confidence score (the highest probability)
    confidence_score = np.max(predictions_proba)
    
    # Get the predicted intent name (RandomForestClassifier.classes_ contains the intent names)
    predicted_intent_name = rf_model.classes_[predicted_index]

    # 3. Apply Confidence Threshold Logic
    if confidence_score < CONFIDENCE_THRESHOLD:
        intent_name = "unrecognized_intent"
        response = RESPONSE_DICT.get(intent_name)
    else:
        intent_name = predicted_intent_name
        # Retrieve the specific response for the predicted intent
        response = RESPONSE_DICT.get(intent_name, RESPONSE_DICT['unrecognized_intent'])

    confidence_display = f"{confidence_score*100:.2f}%"
    
    return intent_name, response, confidence_display


# --- E. Streamlit App Layout (带聊天记录模式) ---
def main():
    st.set_page_config(page_title="RF Intent Chatbot (Chat History)", layout="centered")

    st.title("🌲 Hotel Chatbot (Random Forest)")
    st.caption(f"Confidence Threshold: **{CONFIDENCE_THRESHOLD*100:.0f}%**")

    # 1. 初始化聊天历史 (Session State)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # 增加一个初始的问候消息
        st.session_state.messages.append({"role": "assistant", "content": RESPONSE_DICT['greeting']})

    # 2. 显示聊天历史
    # 使用 st.chat_message 来渲染对话气泡
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # 如果是助手的回复，额外显示置信度和意图
            if message["role"] == "assistant" and "intent" in message:
                st.caption(f"Intent: **{message['intent']}** | Confidence: **{message['confidence']}**")
            st.markdown(message["content"])

    # 3. 处理用户输入
    # 使用 st.chat_input 替换旧的输入和按钮
    user_input = st.chat_input("How may I assist you with your reservation?")
    
    if user_input:
        # 3a. 将用户输入添加到历史记录并显示
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 立即在界面上显示用户输入
        with st.chat_message("user"):
            st.markdown(user_input)

        # 3b. 进行预测并生成回复
        with st.spinner('Analyzing query...'):
            intent_name, response, confidence_display = predict_intent(user_input)
            
            # 3c. 将助手回复添加到历史记录
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "intent": intent_name,
                "confidence": confidence_display
            })

            # 3d. 在界面上显示助手回复
            with st.chat_message("assistant"):
                # 高亮显示意图和置信度
                st.caption(f"Intent: **{intent_name}** | Confidence: **{confidence_display}**")
                st.markdown(response)

if __name__ == "__main__":
    main()
