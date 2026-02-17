import streamlit as st
from streamlit_lottie import st_lottie
import requests, hashlib
from PyPDF2 import PdfReader
from PIL import Image
import torch

from supabase import create_client

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import BlipProcessor, BlipForQuestionAnswering


# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# -------------------- HELPERS --------------------
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None


# -------------------- DATABASE FUNCTIONS --------------------
def register_user(username, password_hash):
    existing = supabase.table("users").select("*").eq("username", username).execute()
    if existing.data:
        return False

    response = supabase.table("users").insert({
        "username": username,
        "password": password_hash,
        "is_admin": False
    }).execute()

    return len(response.data) > 0


def authenticate_user(username, password_hash):
    response = supabase.table("users") \
        .select("*") \
        .eq("username", username) \
        .eq("password", password_hash) \
        .execute()

    if response.data:
        return response.data[0]
    return None


def save_chat(user_id, mode, question, answer):
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "mode": mode,
        "question": question,
        "answer": answer
    }).execute()


def load_chat_history(user_id, mode):
    response = supabase.table("chat_history") \
        .select("question, answer") \
        .eq("user_id", user_id) \
        .eq("mode", mode) \
        .order("id", desc=False) \
        .execute()

    return [(row["question"], row["answer"]) for row in response.data]


def clear_chat_history(user_id, mode):
    supabase.table("chat_history") \
        .delete() \
        .eq("user_id", user_id) \
        .eq("mode", mode) \
        .execute()


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


# -------------------- SESSION INIT --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


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
                user = authenticate_user(u, hash_password(p))
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user["id"]
                    st.session_state.username = user["username"]
                    st.session_state.is_admin = user.get("is_admin", False)
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
st.sidebar.success(f"Logged in as {st.session_state.username} ✅")

if st.sidebar.button("Logout", key="logout_button"):
    for key in ["authenticated", "user_id", "username", "is_admin"]:
        st.session_state[key] = None
    st.rerun()

modes = ["📘 PDF Analyzer", "🖼 Image Q&A"]

if st.session_state.is_admin:
    modes.append("🛠 Admin Dashboard")

mode = st.sidebar.radio("Mode", modes)


# ==================== ADMIN DASHBOARD ====================
if mode == "🛠 Admin Dashboard" and st.session_state.is_admin:

    st.title("🛠 Admin Dashboard")

    users = supabase.table("users").select("*").execute().data
    chats = supabase.table("chat_history").select("*").execute().data

    st.metric("👥 Total Users", len(users))
    st.metric("💬 Total Chats", len(chats))

    st.divider()

    st.subheader("Manage Users")

    for user in users:
        col1, col2 = st.columns([3,1])
        col1.write(user["username"])

        if col2.button("Delete", key=f"del_{user['id']}"):
            supabase.table("users").delete().eq("id", user["id"]).execute()
            supabase.table("chat_history").delete().eq("user_id", user["id"]).execute()
            st.success("User deleted")
            st.rerun()

    st.stop()


# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":

    history = load_chat_history(st.session_state.user_id, "PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.get("current_pdf") != pdf_id:
            st.session_state.current_pdf = pdf_id
            st.session_state.vector_db = None

        if st.session_state.get("vector_db") is None:
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
Use ONLY the context below.

Context:
{context}

Question:
{question}
""")

                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({
                    "context": docs,
                    "question": question
                })

                answer = response.get("output_text", "") \
                    if isinstance(response, dict) else response

                save_chat(st.session_state.user_id, "PDF", question, answer)

            st.rerun()

    for q, a in history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)


# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":

    history = load_chat_history(st.session_state.user_id, "IMAGE")

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

            st.rerun()

    for q, a in history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
import streamlit as st
from streamlit_lottie import st_lottie
import requests, hashlib
from PyPDF2 import PdfReader
from PIL import Image
import torch

from supabase import create_client

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import BlipProcessor, BlipForQuestionAnswering


# -------------------- CONFIG --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# -------------------- HELPERS --------------------
def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None


# -------------------- DATABASE FUNCTIONS --------------------
def register_user(username, password_hash):
    existing = supabase.table("users").select("*").eq("username", username).execute()
    if existing.data:
        return False

    response = supabase.table("users").insert({
        "username": username,
        "password": password_hash,
        "is_admin": False
    }).execute()

    return len(response.data) > 0


def authenticate_user(username, password_hash):
    response = supabase.table("users") \
        .select("*") \
        .eq("username", username) \
        .eq("password", password_hash) \
        .execute()

    if response.data:
        return response.data[0]
    return None


def save_chat(user_id, mode, question, answer):
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "mode": mode,
        "question": question,
        "answer": answer
    }).execute()


def load_chat_history(user_id, mode):
    response = supabase.table("chat_history") \
        .select("question, answer") \
        .eq("user_id", user_id) \
        .eq("mode", mode) \
        .order("id", desc=False) \
        .execute()

    return [(row["question"], row["answer"]) for row in response.data]


def clear_chat_history(user_id, mode):
    supabase.table("chat_history") \
        .delete() \
        .eq("user_id", user_id) \
        .eq("mode", mode) \
        .execute()


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


# -------------------- SESSION INIT --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "is_admin" not in st.session_state:
    st.session_state.is_admin = False


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
                user = authenticate_user(u, hash_password(p))
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user_id = user["id"]
                    st.session_state.username = user["username"]
                    st.session_state.is_admin = user.get("is_admin", False)
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
st.sidebar.success(f"Logged in as {st.session_state.username} ✅")

if st.sidebar.button("Logout"):
    for key in ["authenticated", "user_id", "username", "is_admin"]:
        st.session_state[key] = None
    st.rerun()

modes = ["📘 PDF Analyzer", "🖼 Image Q&A"]

if st.session_state.is_admin:
    modes.append("🛠 Admin Dashboard")

mode = st.sidebar.radio("Mode", modes)


# ==================== ADMIN DASHBOARD ====================
if mode == "🛠 Admin Dashboard" and st.session_state.is_admin:

    st.title("🛠 Admin Dashboard")

    users = supabase.table("users").select("*").execute().data
    chats = supabase.table("chat_history").select("*").execute().data

    st.metric("👥 Total Users", len(users))
    st.metric("💬 Total Chats", len(chats))

    st.divider()

    st.subheader("Manage Users")

    for user in users:
        col1, col2 = st.columns([3,1])
        col1.write(user["username"])

        if col2.button("Delete", key=f"del_{user['id']}"):
            supabase.table("users").delete().eq("id", user["id"]).execute()
            supabase.table("chat_history").delete().eq("user_id", user["id"]).execute()
            st.success("User deleted")
            st.rerun()

    st.stop()


# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":

    history = load_chat_history(st.session_state.user_id, "PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.get("current_pdf") != pdf_id:
            st.session_state.current_pdf = pdf_id
            st.session_state.vector_db = None

        if st.session_state.get("vector_db") is None:
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
Use ONLY the context below.

Context:
{context}

Question:
{question}
""")

                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({
                    "context": docs,
                    "question": question
                })

                answer = response.get("output_text", "") \
                    if isinstance(response, dict) else response

                save_chat(st.session_state.user_id, "PDF", question, answer)

            st.rerun()

    for q, a in history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)


# ==================== IMAGE Q&A ====================
if mode == "🖼 Image Q&A":

    history = load_chat_history(st.session_state.user_id, "IMAGE")

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

            st.rerun()

    for q, a in history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
