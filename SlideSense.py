import streamlit as st
from streamlit_lottie import st_lottie
import requests, hashlib
from PyPDF2 import PdfReader
from PIL import Image
import torch
from supabase import create_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
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

# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
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

# -------------------- AUTH FUNCTIONS --------------------
def register_user(username, password_hash):
    response = supabase.table("users").insert({
        "username": username,
        "password": password_hash
    }).execute()
    return len(response.data) > 0

def authenticate_user(username, password_hash):
    response = supabase.table("users") \
        .select("*") \
        .eq("username", username) \
        .eq("password", password_hash) \
        .execute()
    if response.data:
        return response.data[0]["id"]
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
        .order("id", desc=True) \
        .execute()
    return [(row["question"], row["answer"]) for row in response.data]

# -------------------- SESSION INIT --------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "user_id" not in st.session_state:
    st.session_state.user_id = None

if "username" not in st.session_state:
    st.session_state.username = None

if "current_pdf_id" not in st.session_state:
    st.session_state.current_pdf_id = None

# -------------------- LOGIN UI --------------------
def login_ui():
    col1, col2 = st.columns(2)

    with col1:
        st_lottie(load_lottie(
            "https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
        ), height=300)

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

if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Logged in as {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.clear()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- HERO --------------------
col1, col2 = st.columns([1, 2])
with col1:
    st_lottie(load_lottie(
        "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json"
    ), height=250)
with col2:
    st.markdown("# 📘 SlideSense AI")
    st.markdown("### Production RAG with Supabase")

st.divider()

# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id

            # delete old vectors for user
            supabase.table("documents") \
                .delete() \
                .eq("user_id", st.session_state.user_id) \
                .execute()

            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""

                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=700,
                    chunk_overlap=120
                )

                chunks = splitter.split_text(text)
                embeddings = load_embeddings()

                for chunk in chunks:
                    vector = embeddings.embed_query(chunk)

                    supabase.table("documents").insert({
                        "user_id": st.session_state.user_id,
                        "content": chunk,
                        "embedding": vector
                    }).execute()

        question = st.chat_input("Ask a question about the PDF")

        if question:
            with st.spinner("Thinking..."):
                embeddings = load_embeddings()
                query_vector = embeddings.embed_query(question)

                response = supabase.rpc(
                    "match_documents",
                    {
                        "query_embedding": query_vector,
                        "match_count": 6,
                        "filter_user": st.session_state.user_id
                    }
                ).execute()

                matched_docs = [
                    Document(page_content=row["content"])
                    for row in response.data
                ]

                llm = load_llm()

                prompt = ChatPromptTemplate.from_template("""
You are a precise academic assistant.

Answer strictly using the context below.

Context:
{context}

Question:
{question}

If answer not found say:
"Information not found in the document."
""")

                chain = create_stuff_documents_chain(llm, prompt)

                result = chain.invoke({
                    "context": matched_docs,
                    "question": question
                })

                answer = result.get("output_text", "") \
                    if isinstance(result, dict) else result

                save_chat(
                    st.session_state.user_id,
                    "PDF",
                    question,
                    answer
                )

            st.rerun()

    history = load_chat_history(
        st.session_state.user_id,
        "PDF"
    )

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

            save_chat(
                st.session_state.user_id,
                "IMAGE",
                question,
                answer
            )

            st.rerun()

    history = load_chat_history(
        st.session_state.user_id,
        "IMAGE"
    )

    for q, a in history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)
