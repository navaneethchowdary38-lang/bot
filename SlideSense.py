import streamlit as st
import requests
import hashlib
import json
import time
from datetime import datetime
from PyPDF2 import PdfReader
from PIL import Image
import torch

# Firebase
import firebase_admin
from firebase_admin import credentials, auth, firestore

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import BlipProcessor, BlipForQuestionAnswering


# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SlideSense AI", layout="wide")


# -------------------- FIREBASE INIT --------------------
if not firebase_admin._apps:
    cred_dict = json.loads(st.secrets["FIREBASE_KEY"])
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# -------------------- SESSION INIT --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "email" not in st.session_state:
    st.session_state.email = None
if "mode" not in st.session_state:
    st.session_state.mode = "PDF"


# -------------------- FIREBASE AUTH --------------------
def signup(email, password):
    try:
        user = auth.create_user(email=email, password=password)
        return user
    except:
        return None


def login(email):
    try:
        user = auth.get_user_by_email(email)
        return user
    except:
        return None


# -------------------- FIRESTORE CHAT --------------------
def save_chat(user_id, question, answer, mode):
    db.collection("users") \
      .document(user_id) \
      .collection("chats") \
      .add({
          "question": question,
          "answer": answer,
          "mode": mode,
          "timestamp": datetime.utcnow()
      })


def load_chat_history(user_id, mode):
    chats = db.collection("users") \
              .document(user_id) \
              .collection("chats") \
              .where("mode", "==", mode) \
              .order_by("timestamp") \
              .stream()

    history = []
    for chat in chats:
        data = chat.to_dict()
        history.append((data["question"], data["answer"]))

    return history


# -------------------- CHATGPT STYLE CSS --------------------
st.markdown("""
<style>
.chat-user {
    background-color: #343541;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
}
.chat-ai {
    background-color: #444654;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 15px;
    color: white;
}
.sidebar-title {
    font-size:18px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)


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

        if st.button("Create Account"):
            user = signup(new_email, new_password)
            if user:
                st.success("Account created!")
            else:
                st.error("Signup failed")

    st.stop()


# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Logged in as {st.session_state.email}")

if st.sidebar.button("Logout", key="logout_btn"):
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.email = None
    st.rerun()

mode = st.sidebar.radio(
    "Mode",
    ["📘 PDF Analyzer", "🖼 Image Q&A"]
)

st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"

st.sidebar.markdown('<div class="sidebar-title">💬 Chat History</div>', unsafe_allow_html=True)

history = load_chat_history(st.session_state.user_id, st.session_state.mode)

for i, (q, _) in enumerate(history[-10:]):
    st.sidebar.write(f"• {q[:40]}...")


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )


@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device


# -------------------- PDF ANALYZER --------------------
if st.session_state.mode == "PDF":

    st.title("📘 PDF Analyzer")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:

        if "vector_db" not in st.session_state:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=150
                )

                chunks = splitter.split_text(text)
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        question = st.chat_input("Ask something about the PDF")

        if question:
            docs = st.session_state.vector_db.similarity_search(question, k=8)
            llm = load_llm()

            prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

If not found, say:
Information not found in the document.
""")

            chain = create_stuff_documents_chain(llm, prompt)

            result = chain.invoke({
                "context": docs,
                "question": question
            })

            answer = result.get("output_text", "") \
                if isinstance(result, dict) else result

            save_chat(st.session_state.user_id, question, answer, "PDF")
            st.rerun()

    for q, a in history:
        st.markdown(f'<div class="chat-user">🧑 {q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-ai">🤖 {a}</div>', unsafe_allow_html=True)


# -------------------- IMAGE Q&A --------------------
if st.session_state.mode == "IMAGE":

    st.title("🖼 Image Q&A")

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.chat_input("Ask something about the image")

        if question:
            processor, model, device = load_blip()
            inputs = processor(img, question, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=30)
            answer = processor.decode(outputs[0], skip_special_tokens=True)

            save_chat(st.session_state.user_id, question, answer, "IMAGE")
            st.rerun()

    for q, a in history:
        st.markdown(f'<div class="chat-user">🧑 {q}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-ai">🤖 {a}</div>', unsafe_allow_html=True)
