import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time, hashlib
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
USERS_FILE = "users.json"


# -------------------- HELPERS --------------------
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)


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
defaults = {
    "authenticated": False,
    "users": load_users(),
    "vector_db": None,
    "pdf_chat_history": [],
    "image_chat_history": [],
    "current_pdf_id": None
}


for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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
                if u in st.session_state.users and \
                   st.session_state.users[u] == hash_password(p):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab2:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                if nu in st.session_state.users:
                    st.warning("User already exists")
                else:
                    st.session_state.users[nu] = hash_password(np)
                    save_users(st.session_state.users)
                    st.success("Account created")


# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_length=10, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    try:
        llm = load_llm()

        prompt = f"""
You are an AI assistant.

Question: {question}
Vision Answer: {short_answer}

Convert into one clear and grammatically correct sentence.
Do not add extra information.
"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        # 🔥 Fallback if Gemini quota exceeded
        return f"(Basic Answer) {short_answer}"


# -------------------- AUTH CHECK --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success("Logged in ✅")

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.logged_user = None
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

st.sidebar.markdown("### 💬 Recent Questions")

if mode == "📘 PDF Analyzer":
    history = st.session_state.pdf_chat_history
else:
    history = st.session_state.image_chat_history

if history:
    for q, _ in reversed(history[-5:]):
        st.sidebar.markdown(f"- {q[:40]}...")

    if st.sidebar.button("🧹 Clear Chat History"):
        if mode == "📘 PDF Analyzer":
            st.session_state.pdf_chat_history = []
        else:
            st.session_state.image_chat_history = []
        st.rerun()
else:
    st.sidebar.caption("No history yet")



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
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []

        if st.session_state.vector_db is None:
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
You are a helpful AI assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{question}

If answer is not in context, say:
"Information not found in the document."
""")

                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({
                    "context": docs,
                    "question": question
                })

                answer = response.get("output_text", "") \
                    if isinstance(response, dict) else response

                st.session_state.pdf_chat_history.append((question, answer))

    # -------- CHAT DISPLAY --------
    for q, a in reversed(st.session_state.pdf_chat_history):
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

        question = st.text_input("Ask a question about the image")
        if question:
            with st.spinner("Analyzing image..."):
                st.success(answer_image_question(img, question))
