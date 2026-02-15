import streamlit as st
from streamlit_lottie import st_lottie
import requests, os, time, hashlib, sqlite3
from datetime import datetime
from PyPDF2 import PdfReader
from PIL import Image
import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from transformers import BlipProcessor, BlipForQuestionAnswering


# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

DB_FILE = "slidesense.db"


# -------------------- DATABASE --------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            mode TEXT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


def register_user(username, password_hash):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                  (username, password_hash))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()


def authenticate_user(username, password_hash):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("SELECT id FROM users WHERE username=? AND password=?",
              (username, password_hash))

    result = c.fetchone()
    conn.close()

    return result[0] if result else None


def save_chat(user_id, mode, question, answer):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        INSERT INTO chat_history (user_id, mode, question, answer, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, mode, question, answer, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def load_chat_history(user_id, mode):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        SELECT question, answer
        FROM chat_history
        WHERE user_id=? AND mode=?
        ORDER BY id DESC
    """, (user_id, mode))

    rows = c.fetchall()
    conn.close()
    return rows


def clear_chat_history(user_id, mode):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE user_id=? AND mode=?",
              (user_id, mode))
    conn.commit()
    conn.close()


# -------------------- HELPERS --------------------
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None


# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.4
    )


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(device)
    return processor, model, device


# -------------------- SESSION DEFAULTS --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "username" not in st.session_state:
    st.session_state.username = None

if "user_id" not in st.session_state:
    st.session_state.user_id = None



# -------------------- AUTH UI --------------------
def login_ui():
    col1, col2 = st.columns(2)

    with col1:
        st_lottie(
            load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"),
            height=300
        )

    with col2:
        st.markdown("## 🔐 Welcome to SlideSense")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")

            if st.button("Login"):
                user_id = authenticate_user(u, hash_password(p))
                if user_id:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user_id
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")

            if st.button("Create Account"):
                if register_user(nu, hash_password(np)):
                    st.success("Account created 🎉")
                else:
                    st.warning("User already exists")


# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()


# -------------------- SIDEBAR --------------------
if st.session_state.username:
    st.sidebar.success(f"Logged in as {st.session_state.username} ✅")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

history = load_chat_history(st.session_state.user_id,
                            "PDF" if mode == "📘 PDF Analyzer" else "IMAGE")

if history:
    st.sidebar.markdown("### 💬 Recent Questions")
    for q, _ in history[:5]:
        st.sidebar.markdown(f"- {q[:40]}...")

    if st.sidebar.button("🧹 Clear Chat History"):
        clear_chat_history(st.session_state.user_id,
                           "PDF" if mode == "📘 PDF Analyzer" else "IMAGE")
        st.rerun()


# -------------------- HERO --------------------
col1, col2 = st.columns([1, 2])

with col1:
    st_lottie(
        load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"),
        height=250
    )

with col2:
    st.markdown("# 📘 SlideSense AI")
    st.markdown("### Smart Learning | Smart Vision | Smart AI")

st.divider()


# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if "vector_db" not in st.session_state:
            with st.spinner("📄 Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""

                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,
                    chunk_overlap=60
                )

                chunks = splitter.split_text(text)
                embeddings = load_embeddings()
                st.session_state.vector_db = FAISS.from_texts(
                    chunks, embeddings
                )

        question = st.chat_input("Ask a question about the PDF")

        if question:
            with st.spinner("🤖 Thinking..."):
                docs = st.session_state.vector_db.similarity_search(question, k=4)

                llm = load_llm()

                prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

If answer is not found, say:
"Information not found in the document."
""")

                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({
                    "context": docs,
                    "question": question
                })

                answer = response.get("output_text", "") \
                    if isinstance(response, dict) else response

                save_chat(st.session_state.user_id, "PDF", question, answer)

    for q, a in history:
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            st.markdown(a)


# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":

    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.chat_input("Ask a question about the image")

        if question:
            processor, model, device = load_blip()
            inputs = processor(img, question, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_length=20, num_beams=5)
            answer = processor.decode(outputs[0], skip_special_tokens=True)

            save_chat(st.session_state.user_id, "IMAGE", question, answer)

    for q, a in history:
        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            st.markdown(a)
