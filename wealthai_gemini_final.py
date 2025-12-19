# final_gemini_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION (This MUST be the absolute first Streamlit command)
# ---------------------------------------------------------
st.set_page_config(page_title="WealthAI Copilot", page_icon="🤝", layout="wide")

# ---------------------------------------------------------
# 2. AUTOMATIC MODEL FINDER
# ---------------------------------------------------------
# PASTE YOUR KEY HERE
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 

def configure_gemini():
    """Tries to find a working model automatically."""
    try:
        # Check for placeholder key
        if "YOUR_ACTUAL" in GEMINI_API_KEY:
            st.error("⚠️ Please replace 'YOUR_ACTUAL_API_KEY_HERE' with your real Gemini API key in the code.")
            return None

        genai.configure(api_key=GEMINI_API_KEY)
        
        # 1. Try to list models available to this API key
        available_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except Exception as e:
            # If listing fails, we can't do dynamic selection
            st.warning(f"Could not list models: {e}. Defaulting to 'gemini-pro'.")
            return genai.GenerativeModel('gemini-pro')

        # 2. Pick the best model from the list
        if not available_models:
            st.error("No models found for this API Key. Check your Google AI Studio account permissions.")
            return None
        
        # Prefer gemini-1.5-flash (fast), then gemini-pro (stable)
        selected_model_name = ""
        if 'models/gemini-1.5-flash' in available_models:
            selected_model_name = 'gemini-1.5-flash'
        elif 'models/gemini-pro' in available_models:
            selected_model_name = 'gemini-pro'
        else:
            # Just take the first one available
            selected_model_name = available_models[0]
            
        print(f"DEBUG: Using model -> {selected_model_name}")
        return genai.GenerativeModel(selected_model_name)

    except Exception as e:
        st.error(f"Critical Error configuring Gemini: {e}")
        return None

# Initialize the model once
model = configure_gemini()

# -------------------------
# 3. CSS STYLING
# -------------------------
def local_css():
    st.markdown("""
    <style>
        .stButton > button { border-radius: 12px; font-weight: 600; }
        .copilot-header { text-align: center; background: #f8f9fa; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        .copilot-title { font-size: 28px; font-weight: 800; color: #1a1a1a; }
        .stChatMessage { padding: 1rem; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# 4. RAG LOGIC (Loading Data)
# -------------------------
@st.cache_resource
def load_embedding_model():
    # Load model once and cache it
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_knowledge_base():
    try:
        with open('industry_project_df.pkl', 'rb') as f:
            df = pickle.load(f)
        
        # Check if embeddings exist; if not, generate them
        if 'embedding' not in df.columns:
            # We use Python print instead of st.write to avoid UI structure errors
            print("Generating embeddings... this may take a moment.")
            embed_model = load_embedding_model()
            df['combined_text'] = df['title'].astype(str) + ": " + df['content'].astype(str)
            df['embedding'] = df['combined_text'].apply(lambda x: embed_model.encode(x))
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading PKL file: {e}")
        return pd.DataFrame()

# Initialize Global Resources
embedder = load_embedding_model()
knowledge_base = load_knowledge_base()

def retrieve_context(query, top_k=3):
    """Retrieves best 3 matches from the dataframe."""
    if knowledge_base.empty:
        return ""
    
    # Encode query
    query_embedding = embedder.encode(query).reshape(1, -1)
    
    try:
        # Stack embeddings
        corpus_embeddings = np.vstack(knowledge_base['embedding'].values)
    except ValueError:
        return ""
    
    # Calculate similarity
    similarities = cosine_similarity(query_embedding, corpus_embeddings)
    
    # Get top k
    top_indices = np.argsort(similarities[0])[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        row = knowledge_base.iloc[idx]
        results.append(f"Title: {row['title']}\nContent: {row['content']}")
    
    return "\n\n".join(results)

def query_gemini(prompt, context):
    """Queries the Gemini API with error handling."""
    if not model:
        return "⚠️ API Error: No working Gemini model found. Did you include 'google-generativeai>=0.4.0' in requirements.txt?"

    system_prompt = (
        "You are a helpful banking relationship manager assistant. "
        "Use the provided context to answer the user's question accurately. "
        "If the answer is not in the context, use your general knowledge but mention strictly that it is general info."
    )
    
    full_prompt = f"""
    System Instruction: {system_prompt}
    
    Context Information:
    {context}
    
    User Question:
    {prompt}
    """
    
    try:
        # Generate content using Gemini
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Gemini Error: {e}"

# -------------------------
# 5. MAIN APP LOGIC
# -------------------------

# Session State Setup
if "users" not in st.session_state:
    st.session_state.users = {"john@bank.com": {"password": "12345", "name": "John D."}}
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "Auth"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "dashboard_view" not in st.session_state: st.session_state.dashboard_view = "main"
if "selected_activity" not in st.session_state: st.session_state.selected_activity = None
if "auth_mode" not in st.session_state: st.session_state.auth_mode = "login"

if "recent_activities" not in st.session_state:
    st.session_state.recent_activities = [
        {"id": "act_001", "type": "Meeting", "title": "Portfolio Review", "time": "10:00 AM", "details": "Discussed Q3 performance."},
        {"id": "act_002", "type": "Compliance", "title": "LRS Declaration", "time": "09:15 AM", "details": "Submitted Form A2."}
    ]

def handle_new_message(prompt):
    st.session_state.chat_history.append({"role": "user", "text": prompt})
    st.session_state.page = "Chat" # Force switch to chat
    
    with st.spinner("Thinking (Gemini)..."):
        context = retrieve_context(prompt)
        response = query_gemini(prompt, context)
        st.session_state.chat_history.append({"role": "assistant", "text": response})

# --- UI Functions ---
def render_sidebar():
    local_css()
    with st.sidebar:
        st.write("👤 **John D.** (Senior RM)")
        st.divider()
        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.session_state.dashboard_view = "main"
            st.rerun()
        if st.button("💬 AI Chat", use_container_width=True, type="primary"):
            st.session_state.page = "Chat"
            st.rerun()
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "Auth"
            st.rerun()

def auth_screen():
    local_css()
    col1, col2 = st.columns([6,4])
    with col1: st.image("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800", use_container_width=True)
    with col2:
        st.title("WealthAI Login")
        with st.form("login"):
            email = st.text_input("Email", "john@bank.com")
            pwd = st.text_input("Password", "12345", type="password")
            if st.form_submit_button("Login", type="primary", use_container_width=True):
                if email in st.session_state.users and st.session_state.users[email]["password"] == pwd:
                    st.session_state.logged_in = True
                    st.session_state.page = "Dashboard"
                    st.rerun()
                else: st.error("Invalid credentials")

def dashboard_screen():
    render_sidebar()
    if st.session_state.dashboard_view == "details_activity" and st.session_state.selected_activity:
        act = st.session_state.selected_activity
        st.subheader("📝 Activity Details")
        st.button("← Back", on_click=lambda: st.session_state.update(dashboard_view="main"))
        st.info(f"**{act['title']}**\n\n{act['details']}")
        return

    st.title("Dashboard")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.success("🚀 **AI Copilot Online (Gemini)**")
        st.markdown("Use the chat to ask about **NRE Accounts, FEMA Guidelines, or Investment Limits**.")
        if st.button("✨ Launch AI Chat", type="primary"):
            st.session_state.page = "Chat"
            st.rerun()
    with c2:
        st.subheader("Recent Activity")
        for act in st.session_state.recent_activities:
            if st.button(f"{act['time']} - {act['title']}", key=act['id'], use_container_width=True):
                st.session_state.selected_activity = act
                st.session_state.dashboard_view = "details_activity"
                st.rerun()

def chat_screen():
    render_sidebar()
    st.markdown("<div class='copilot-header'><h3 class='copilot-title'>WealthAI Copilot</h3></div>", unsafe_allow_html=True)
    if knowledge_base.empty:
        st.warning("⚠️ Knowledge Base (`industry_project_df.pkl`) not found or empty.")
    
    with st.container():
        if not st.session_state.chat_history:
            st.info("👋 Hello! Ask me about banking compliance or client portfolios.")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
                
    if prompt := st.chat_input("Ask a question..."):
        handle_new_message(prompt)
        st.rerun()

if not st.session_state.logged_in:
    auth_screen()
elif st.session_state.page == "Dashboard":
    dashboard_screen()
elif st.session_state.page == "Chat":
    chat_screen()