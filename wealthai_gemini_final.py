# final_gemini_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ---------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="WealthAI Copilot", page_icon="🤝", layout="wide")

# ---------------------------------------------------------
# 2. IDEAL PROMPT DOCUMENTATION (FRONTEND GUIDE)
# ---------------------------------------------------------
IDEAL_PROMPT_DOC = """
### ✅ How to Ask Effective Questions

To get the most accurate answers, frame your questions clearly.

**Best Practices**
- Clearly mention the topic (Taxation, FEMA, NRE/NRO, LRS, Investments)
- Ask one question at a time
- Keep it factual and compliance-oriented

**Good Examples**
- What is the taxation on NRE fixed deposit interest?
- Explain capital gains tax for NRIs investing in mutual funds
- What are FEMA guidelines for outward remittance under LRS?
- Is TDS applicable on NRO account interest?

**Avoid**
- Vague questions (e.g. "Tell me about tax")
- Multiple questions in one prompt
- Personal financial advice questions

📌 *Responses are generated from the internal knowledge base. If information is missing, the system will notify you.*
"""

# ---------------------------------------------------------
# 3. GEMINI CONFIGURATION
# ---------------------------------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")

def configure_gemini():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        models = []
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                models.append(m.name)

        if "models/gemini-1.5-flash" in models:
            return genai.GenerativeModel("gemini-1.5-flash")
        elif "models/gemini-pro" in models:
            return genai.GenerativeModel("gemini-pro")
        elif models:
            return genai.GenerativeModel(models[0])
    except Exception as e:
        st.error(f"Gemini configuration failed: {e}")
    return None

model = configure_gemini()

# ---------------------------------------------------------
# 4. STYLING
# ---------------------------------------------------------
def local_css():
    st.markdown("""
    <style>
        .stButton > button { border-radius: 12px; font-weight: 600; }
        .copilot-header { text-align: center; background: #f8f9fa; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        .copilot-title { font-size: 28px; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. RAG LOGIC
# ---------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_knowledge_base():
    rows = []

    try:
        with open("industry_project_df.pkl", "rb") as f:
            df_old = pickle.load(f)
        for _, r in df_old.iterrows():
            rows.append({
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "embedding": r.get("embedding")
            })
    except Exception:
        pass

    if os.path.exists("knowledge"):
        for file in os.listdir("knowledge"):
            if file.endswith(".txt"):
                with open(os.path.join("knowledge", file), "r", encoding="utf-8") as f:
                    rows.append({
                        "title": file,
                        "content": f.read(),
                        "embedding": None
                    })

    df = pd.DataFrame(rows)

    if not df.empty:
        embedder = load_embedding_model()
        mask = df["embedding"].isnull()
        if mask.any():
            df.loc[mask, "embedding"] = df.loc[mask].apply(
                lambda x: embedder.encode(f"{x['title']} {x['content']}"),
                axis=1
            )

    return df.dropna(subset=["embedding"])

embedder = load_embedding_model()
knowledge_base = load_knowledge_base()

def retrieve_context(query, top_k=3):
    if knowledge_base.empty:
        return ""
    q_emb = embedder.encode(query).reshape(1, -1)
    corpus = np.vstack(knowledge_base["embedding"].values)
    sims = cosine_similarity(q_emb, corpus)[0]
    idx = sims.argsort()[-top_k:][::-1]
    return "\n\n".join(
        f"Title: {knowledge_base.iloc[i]['title']}\n{knowledge_base.iloc[i]['content']}"
        for i in idx
    )

def check_taxation_rag():
    test_queries = [
        "taxation",
        "capital gains tax",
        "income tax for NRI",
        "TDS on NRO account"
    ]
    for q in test_queries:
        ctx = retrieve_context(q)
        if ctx and len(ctx.strip()) > 50:
            return True, q, ctx[:300]
    return False, None, None

def query_gemini(prompt, context):
    if not model:
        return "Gemini model not available."

    if not context.strip():
        return (
            "⚠️ No relevant information found in the knowledge base for this question. "
            "Please verify documents or rephrase the query."
        )

    full_prompt = f"""
You are a banking relationship manager assistant.
Answer strictly using the provided context.

Context:
{context}

Question:
{prompt}
"""
    try:
        return model.generate_content(full_prompt).text
    except Exception as e:
        return f"Gemini error: {e}"

# ---------------------------------------------------------
# 6. SESSION STATE + DUMMY AUTH
# ---------------------------------------------------------
DUMMY_USERS = {
    "rm01": {"password": "rm01@123", "name": "John D.", "role": "Senior RM"},
    "rm02": {"password": "rm02@123", "name": "Alice K.", "role": "Relationship Manager"}
}

if "users" not in st.session_state:
    st.session_state.users = DUMMY_USERS
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Auth"
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# 7. UI SCREENS
# ---------------------------------------------------------
def auth_screen():
    local_css()
    st.title("🔐 WealthAI Login")

    with st.form("login"):
        uid = st.text_input("User ID", "rm01")
        pwd = st.text_input("Password", "rm01@123", type="password")
        if st.form_submit_button("Login", type="primary"):
            if uid in st.session_state.users and st.session_state.users[uid]["password"] == pwd:
                st.session_state.logged_in = True
                st.session_state.current_user = st.session_state.users[uid]
                st.session_state.page = "Dashboard"
                st.rerun()
            else:
                st.error("Invalid User ID or Password")

def render_sidebar():
    local_css()
    with st.sidebar:
        u = st.session_state.current_user or {}
        st.write(f"👤 **{u.get('name','User')}** ({u.get('role','RM')})")
        st.divider()

        if st.button("🏠 Dashboard", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()
        if st.button("💬 AI Chat", use_container_width=True):
            st.session_state.page = "Chat"
            st.rerun()

        st.divider()
        with st.expander("📝 Ideal Prompt Guide"):
            st.markdown(IDEAL_PROMPT_DOC)

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.page = "Auth"
            st.rerun()

def dashboard_screen():
    render_sidebar()
    st.title("Dashboard")
    st.success("🚀 AI Copilot Online")

    with st.expander("🧪 Knowledge Check: Taxation"):
        ok, q, sample = check_taxation_rag()
        if ok:
            st.success("✅ Taxation knowledge is WORKING")
            st.caption(f"Matched query: **{q}**")
            st.code(sample)
        else:
            st.error("❌ Taxation knowledge NOT found in knowledge base")

def chat_screen():
    render_sidebar()
    st.markdown(
        "<div class='copilot-header'><div class='copilot-title'>WealthAI Copilot</div></div>",
        unsafe_allow_html=True
    )

    if not st.session_state.chat_history:
        with st.expander("🧭 How to Ask Questions", expanded=True):
            st.markdown(IDEAL_PROMPT_DOC)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.chat_history.append({"role": "user", "text": prompt})
        ctx = retrieve_context(prompt)
        resp = query_gemini(prompt, ctx)
        st.session_state.chat_history.append({"role": "assistant", "text": resp})
        st.rerun()

# ---------------------------------------------------------
# 8. ROUTER
# ---------------------------------------------------------
if not st.session_state.logged_in:
    auth_screen()
elif st.session_state.page == "Dashboard":
    dashboard_screen()
elif st.session_state.page == "Chat":
    chat_screen()
