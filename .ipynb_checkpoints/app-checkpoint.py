# app.py

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

try:
    # 这些变量现在是全局的，并且在脚本开始时就被定义了
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    # 如果下载失败，我们在这里可以捕获并提供错误
    st.error("NLTK data (stopwords, wordnet) not found. Please ensure resources are downloaded and accessible.")
    st.stop()

# --- NLTK 资源加载函数 (使用 Streamlit 缓存) ---
@st.cache_resource
def load_nltk_data():
    """Download NLTK resources once and initialize tools."""
    try:
        # 显式下载所有必需的资源
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        return stop_words, lemmatizer
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        st.stop()

# 在应用启动时调用一次
load_nltk_data()

# --- 1. Constants and Initial Setup ---
MODEL_PATH = 'naive_bayes_intent_model.joblib'
VECTORIZER_PATH = 'tfidf_vectorizer.joblib'
DATASET_PATH = 'dataset.csv' # 新增：数据集路径

# --- 2. Load Model, Vectorizer, and Data ---

@st.cache_resource # 缓存资源以避免在每次重运行时重复加载
def load_resources():
    """加载保存的模型、向量化器和数据集。"""
    model, vectorizer, df = None, None, None
    try:
        model = load(MODEL_PATH)
        vectorizer = load(VECTORIZER_PATH)
    except FileNotFoundError:
        st.error(f"Error: Could not find model or vectorizer files. Please ensure '{MODEL_PATH}' and '{VECTORIZER_PATH}' are present.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()
        
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        st.warning(f"Warning: Could not find dataset file '{DATASET_PATH}'. Quick query buttons will be disabled.")
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        
    return model, vectorizer, df

nb_model, vectorizer, df_data = load_resources()

def preprocess_text(text):
    """Applies the same NLTK preprocessing steps as used during training."""
    # 1. Convert to Lowercase
    text = text.lower()
    
    # 2. Remove Punctuation and Special Characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Lemmatization (Key Enhancement)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Rejoin tokens into a single string
    return ' '.join(tokens)

# --- 3. Predefined Responses ---

responses = {
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

# --- 4. Chatbot Logic Function (Same as before) ---

def chatbot_reply_nb(user_input, model, vectorizer, responses):
    """根据用户输入预测意图并返回相应回复。"""
    if not user_input.strip():
        return "Please enter a question to start the conversation.", "Empty Input", 0.0

    processed_input = preprocess_text(user_input)
    vector = vectorizer.transform([processed_input])
    probabilities = model.predict_proba(vector)[0]
    intent_index = np.argmax(probabilities)
    confidence = probabilities[intent_index]
    intent = model.classes_[intent_index]
    
    CONFIDENCE_THRESHOLD = 0.3
    
    if confidence < CONFIDENCE_THRESHOLD:
        reply = f"Sorry, I'm not sure I understand. My predicted intent ('{intent}') had a low confidence score ({confidence:.2f}). Could you please rephrase?"
        predicted_intent = "Fallback (Low Confidence)"
    else:
        reply = responses.get(intent, f"Sorry, I predicted the intent **'{intent}'** (Confidence: {confidence:.2f}), but I don't have a specific response for that yet. Please rephrase your question.")
        predicted_intent = intent

    return reply, predicted_intent, confidence

# --- 5. Core Chat Function (Handles the interaction flow) ---

def handle_chat_interaction(prompt):
    """处理用户输入、更新聊天历史并生成回复。"""
    # 1. 存储用户消息到历史
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. 获取聊天机器人回复
    reply, predicted_intent, confidence = chatbot_reply_nb(prompt, nb_model, vectorizer, responses)
    
    # 3. 存储机器人消息到历史
    st.session_state.messages.append({"role": "assistant", "content": reply, "intent": predicted_intent, "confidence": confidence})
    
    # 4. 强制重新运行以显示新的历史消息
    # Streamlit 通常会自己刷新，但这个 pattern 在某些情况下更可靠
    st.rerun()


# --- 6. Streamlit UI Setup ---

st.set_page_config(page_title="Intent-Based Chatbot Demo", layout="centered")

st.title("🛎️ Intent-Based Chatbot Demo")
st.markdown("Powered by **Multinomial Naive Bayes** and **TF-IDF**.")

# 初始化对话历史（使用 session state）
if "messages" not in st.session_state:
    st.session_state.messages = []
    # 初始欢迎语
    initial_response = responses["greeting"]
    st.session_state.messages.append({"role": "assistant", "content": initial_response, "intent": "greeting", "confidence": 1.0})

# --- Quick Query Buttons ---
if df_data is not None and not df_data.empty:
    st.markdown("---")
    st.subheader("🚀 Quick Queries from Dataset")
    
    # 从数据集中随机抽取最多 5 个样本
    quick_queries = df_data['text'].sample(min(5, len(df_data)), random_state=42).tolist()
    
    # 使用 st.columns 或 st.button 来创建按钮布局
    cols = st.columns(len(quick_queries))
    for i, query in enumerate(quick_queries):
        if cols[i].button(query):
            # 当按钮被点击时，调用处理函数
            handle_chat_interaction(query)

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "intent" in message and message["intent"] != "greeting":
                st.caption(f"**Predicted Intent:** {message['intent']} | **Confidence:** {message['confidence']:.2f}")

# --- User Input Text Box ---
if prompt := st.chat_input("Ask a question about the hotel:"):
    handle_chat_interaction(prompt)

