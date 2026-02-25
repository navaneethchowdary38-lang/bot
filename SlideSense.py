import streamlit as st
from datetime import datetime
import uuid
from PyPDF2 import PdfReader
from PIL import Image
import base64

# Firebase
import firebase_admin
from firebase_admin import credentials, auth, firestore

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense AI", layout="wide")

# -------------------- FIREBASE INIT --------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(st.secrets["firebase"])
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------- SESSION --------------------
defaults = {
    "authenticated": False,
    "user_id": None,
    "email": None,
    "mode": "PDF",
    "current_chat_id": None,
    "pdf_chunks": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- AUTH --------------------
def signup(email, password):
    try:
        return auth.create_user(email=email, password=password)
    except:
        return None

def login(email):
    try:
        return auth.get_user_by_email(email)
    except:
        return None

# -------------------- CHAT DB --------------------
def create_new_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    db.collection("users").document(user_id)\
      .collection("chats").document(chat_id)\
      .set({
          "mode": mode,
          "created_at": datetime.utcnow(),
          "title": "New Chat"
      })
    return chat_id

def save_message(user_id, chat_id, role, content):
    db.collection("users")\
      .document(user_id)\
      .collection("chats")\
      .document(chat_id)\
      .collection("messages")\
      .add({
          "role": role,
          "content": content,
          "timestamp": datetime.utcnow()
      })

def load_messages(user_id, chat_id):
    docs = db.collection("users")\
             .document(user_id)\
             .collection("chats")\
             .document(chat_id)\
             .collection("messages")\
             .order_by("timestamp")\
             .stream()
    return [(d.to_dict()["role"], d.to_dict()["content"]) for d in docs]

# -------------------- LOGIN UI --------------------
if not st.session_state.authenticated:
    st.title("🔐 SlideSense Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = login(email)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user.uid
                st.session_state.email = email
                st.rerun()
            else:
                st.error("User not found")

    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")
        if st.button("Signup"):
            user = signup(new_email, new_password)
            if user:
                st.success("Account created!")

    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Logged in as {st.session_state.email}")

if st.sidebar.button("Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"

if st.sidebar.button("➕ New Chat"):
    cid = create_new_chat(st.session_state.user_id, st.session_state.mode)
    st.session_state.current_chat_id = cid
    st.session_state.pdf_chunks = None
    st.rerun()

# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

# -------------------- MAIN --------------------
if st.session_state.current_chat_id:

    llm = load_llm()

    if st.session_state.mode == "PDF":
        st.title("📘 PDF Analyzer")

        pdf = st.file_uploader("Upload PDF", type="pdf")

        if pdf and st.session_state.pdf_chunks is None:
            with st.spinner("Reading PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                st.session_state.pdf_chunks = splitter.split_text(text)

    if st.session_state.mode == "IMAGE":
        st.title("🖼 Image Q&A")
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    # Load old messages
    messages = load_messages(
        st.session_state.user_id,
        st.session_state.current_chat_id
    )

    for role, content in messages:
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

    question = st.chat_input("Ask something...")

    if question:
        save_message(
            st.session_state.user_id,
            st.session_state.current_chat_id,
            "user",
            question
        )

        # ---------------- PDF MODE ----------------
        if st.session_state.mode == "PDF":
            if not st.session_state.pdf_chunks:
                answer = "Please upload a PDF first."
            else:
                context = "\n\n".join(
                    st.session_state.pdf_chunks[:6]
                )

                prompt = f"""
You are a PDF assistant.

Context:
{context}

Question:
{question}

If answer not in context say:
Information not found in document.
"""
                response = llm.invoke(prompt)
                answer = response.content

        # ---------------- IMAGE MODE ----------------
        else:
            if not img_file:
                answer = "Please upload an image first."
            else:
                image = Image.open(img_file)

                response = llm.invoke([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{base64.b64encode(img_file.getvalue()).decode()}"
                            }
                        ]
                    }
                ])
                answer = response.content

        save_message(
            st.session_state.user_id,
            st.session_state.current_chat_id,
            "assistant",
            answer
        )

        st.rerun()

else:
    st.title("🚀 Start a New Chat from Sidebar")
