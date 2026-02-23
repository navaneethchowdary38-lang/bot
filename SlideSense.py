import streamlit as st
from streamlit_lottie import st_lottie
import requests, json, os, time
from typing import Any, Dict, Optional, List

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

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    st.error(
        "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_ANON_KEY "
        "environment variables before running this app."
    )
    st.stop()


# -------------------- AUTH HELPERS (Supabase HTTP) --------------------
def _auth_headers(access_token: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Content-Type": "application/json",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return headers


def _auth_request(method: str, path: str, **kwargs) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    return requests.request(method, url, headers=_auth_headers(), timeout=10, **kwargs)


def _db_request(method: str, path: str, access_token: str, **kwargs) -> requests.Response:
    url = f"{SUPABASE_URL}{path}"
    return requests.request(
        method,
        url,
        headers=_auth_headers(access_token),
        timeout=10,
        **kwargs,
    )


def set_session(access_token: str, refresh_token: str, user: Dict[str, Any]) -> None:
    st.session_state["session"] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": user,
    }


def current_user() -> Optional[Dict[str, Any]]:
    sess = st.session_state.get("session")
    if not sess:
        return None
    return sess["user"]


def _current_token() -> Optional[str]:
    sess = st.session_state.get("session")
    if not sess:
        return None
    return sess["access_token"]


def sign_up(email: str, password: str) -> Optional[str]:
    payload = {"email": email, "password": password}
    resp = _auth_request("POST", "/auth/v1/signup", json=payload)
    if resp.status_code >= 400:
        return resp.text
    return None


def sign_in(email: str, password: str) -> Optional[str]:
    payload = {"email": email, "password": password}
    resp = _auth_request(
        "POST",
        "/auth/v1/token?grant_type=password",
        json=payload,
    )
    if resp.status_code >= 400:
        return resp.text

    data = resp.json()
    access = data.get("access_token")
    refresh = data.get("refresh_token")
    user = data.get("user")
    if not access or not user:
        return "Invalid auth response from Supabase."

    set_session(access, refresh, user)
    return None


def sign_out() -> None:
    st.session_state.pop("session", None)


# -------------------- HELPERS --------------------
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None


def type_text(text, speed=0.03):
    box = st.empty()
    out = ""
    for c in text:
        out += c
        box.markdown(f"### {out}")
        time.sleep(speed)


# -------------------- CACHED MODELS --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


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
    "vector_db": None,
    "chat_history": [],
    "current_pdf_id": None,
    "guest": False,
    "history_loaded": False,
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
        type_text("🔐 Welcome to SlideSense")

        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Guest"])

        with tab1:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if not email or not password:
                    st.warning("Enter email and password.")
                else:
                    err = sign_in(email, password)
                    if err:
                        st.error(f"Login failed: {err}")
                    else:
                        st.rerun()

        with tab2:
            email = st.text_input("Email", key="signup_email")
            password = st.text_input(
                "Password (min 6 chars)", type="password", key="signup_password"
            )
            if st.button("Create Account"):
                if not email or not password:
                    st.warning("Enter email and password.")
                else:
                    err = sign_up(email, password)
                    if err:
                        st.error(f"Sign-up failed: {err}")
                    else:
                        st.success(
                            "Check your email to verify your account before logging in."
                        )

        with tab3:
            st.markdown("Continue without creating an account.")
            if st.button("Continue as guest"):
                st.session_state["guest"] = True
                st.rerun()


# -------------------- CHAT HISTORY PERSISTENCE (Supabase) --------------------
def load_chat_history_from_db() -> None:
    """Load PDF chat history for the current user from Supabase into session_state."""
    if st.session_state.get("guest"):
        return
    user = current_user()
    token = _current_token()
    if not user or not token:
        return

    resp = _db_request(
        "GET",
        "/rest/v1/chat_history"
        "?select=question,answer,mode,created_at"
        f"&user_id=eq.{user['id']}"
        "&mode=eq.pdf"
        "&order=created_at.asc",
        access_token=token,
    )
    if resp.status_code >= 400:
        return

    rows: List[Dict[str, Any]] = resp.json()
    st.session_state.chat_history = [(r["question"], r["answer"]) for r in rows]
    st.session_state.history_loaded = True


def save_chat_to_db(question: str, answer: str) -> None:
    """Append a single PDF QA pair to Supabase chat_history."""
    if st.session_state.get("guest"):
        return
    user = current_user()
    token = _current_token()
    if not user or not token:
        return

    payload = {
        "user_id": user["id"],
        "mode": "pdf",
        "question": question,
        "answer": answer,
    }
    _db_request(
        "POST",
        "/rest/v1/chat_history",
        access_token=token,
        json=payload,
    )

# -------------------- IMAGE Q&A --------------------
def answer_image_question(image, question):
    processor, model, device = load_blip()
    inputs = processor(image, question, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_length=10, num_beams=5)
    short_answer = processor.decode(outputs[0], skip_special_tokens=True)

    llm = load_llm()
    prompt = f"""
Question: {question}
Vision Answer: {short_answer}
Convert into one clear sentence. No extra details.
"""
    return llm.invoke(prompt).content

# -------------------- AUTH CHECK --------------------
user = current_user()
if (not user) and not st.session_state.get("guest"):
    login_ui()
    st.stop()

# -------------------- SIDEBAR --------------------
user = current_user()
label = (
    f"Logged in as {user.get('email')}" if user else "Logged in as Guest"
    if st.session_state.get("guest")
    else "Not logged in"
)
st.sidebar.success(label)

if st.sidebar.button("Logout"):
    st.cache_resource.clear()
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.session_state["guest"] = False
    sign_out()
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

# -------------------- SIDEBAR HISTORY --------------------
st.sidebar.markdown("### 💬 Chat History")

if st.session_state.chat_history:
    # Latest first in the sidebar list
    items = list(reversed(list(enumerate(st.session_state.chat_history, start=1))))
    labels = [f"{idx}. {q[:40]}..." for idx, (q, _) in items]

    selected_label = st.sidebar.selectbox(
        "Select a message",
        options=labels,
        label_visibility="collapsed",
        key="history_select",
    )

    # Map selection back to the corresponding Q/A pair
    if selected_label:
        sel_idx = int(selected_label.split(".")[0])
        q_sel, a_sel = st.session_state.chat_history[sel_idx - 1]

        with st.sidebar.expander("Selected chat", expanded=True):
            st.markdown("**You**")
            st.write(q_sel)
            st.markdown("**Assistant**")
            st.write(a_sel)

    if st.sidebar.button("🧹 Clear History"):
        st.session_state.chat_history = []
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
    type_text("📘 SlideSense AI Platform")
    st.markdown("### Smart Learning | Smart Vision | Smart AI")

st.divider()

# ==================== PDF ANALYZER ====================
if mode == "📘 PDF Analyzer":
    pdf = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    if pdf:
        pdf_id = f"{pdf.name}_{pdf.size}"

        if st.session_state.current_pdf_id != pdf_id:
            st.session_state.current_pdf_id = pdf_id
            st.session_state.vector_db = None
            st.session_state.chat_history = []

        if (not st.session_state.get("history_loaded")) and (not st.session_state.get("guest")):
            load_chat_history_from_db()

        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""

                for pdf_page in reader.pages:
                    extracted = pdf_page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                if not text.strip():
                    st.error("No readable text found in PDF")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )
                chunks = splitter.split_text(text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        q = st.text_input("Ask a question")

        if q:
            llm = load_llm()
            docs = st.session_state.vector_db.similarity_search(q, k=5)

            prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context": docs, "question": q})

            if isinstance(res, dict):
                answer = res.get("output_text", "")
            else:
                answer = res

            st.session_state.chat_history.append((q, answer))
            save_chat_to_db(q, answer)

        # -------- CHAT DISPLAY (ChatGPT-style, latest on top) --------
        st.markdown("## 💬 Conversation")

        chat_container = st.container()
        with chat_container:
            for uq, ua in reversed(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.markdown(uq)
                with st.chat_message("assistant"):
                    st.markdown(ua)

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
