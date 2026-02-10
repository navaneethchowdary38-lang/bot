import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import time

# -------------------- Page Config --------------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

# -------------------- Lottie Loader --------------------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

login_anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
#ai_anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json")
upload_anim = load_lottie("https://assets10.lottiefiles.com/packages/lf20_ysrn2iwp.json")

# -------------------- Session Defaults --------------------
defaults = {
    "chat_history": [],
    "vector_db": None,
    "authenticated": False,
    "users": {"admin": "admin123"}
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- Typing Effect --------------------
def type_text(text, speed=0.03):
    box = st.empty()
    typed = ""
    for c in text:
        typed += c
        box.markdown(f"### {typed}")
        time.sleep(speed)

# -------------------- Authentication UI --------------------
def login_ui():
    col1, col2 = st.columns(2)

    with col1:
        st_lottie(login_anim, height=300)

    with col2:
        type_text("🔐 Welcome to SlideSense")
        st.markdown("### AI Powered Learning Platform")

        tab1, tab2 = st.tabs(["Login", "Sign Up"])

        with tab1:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login"):
                if u in st.session_state.users and st.session_state.users[u] == p:
                    st.success("Login Successful 🚀")
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials ❌")

        with tab2:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            if st.button("Create Account"):
                if nu in st.session_state.users:
                    st.warning("User already exists")
                else:
                    st.session_state.users[nu] = np
                    st.success("Account created 🎉")

# -------------------- Load BLIP VQA --------------------
@st.cache_resource
def load_blip_vqa():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    return processor, model

processor, blip_vqa_model = load_blip_vqa()

# -------------------- UPDATED IMAGE Q&A (OPTION-1) --------------------
def answer_image_question(image, question):
    # Step 1: BLIP short answer
    inputs = processor(image, question, return_tensors="pt")

    output = blip_vqa_model.generate(
        **inputs,
        max_length=10,
        num_beams=5,
        early_stopping=True
    )

    short_answer = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    # Step 2: Expand using Gemini (text-only)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    expansion_prompt = f"""
You are expanding a visual answer.

Image Question:
{question}

Vision Model Answer:
{short_answer}

Task:
- Convert this into a clear, complete sentence.
- Do NOT add extra details.
- Keep accurate.
"""

    final_answer = llm.invoke(expansion_prompt)
    return final_answer.content

# -------------------- Auth Check --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.success("Logged in ✅")
if st.sidebar.button("Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

page = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])

st.sidebar.markdown("### 💬 History")
for q, a in st.session_state.chat_history[-6:]:
    st.sidebar.markdown(f"- {q[:30]}")

# -------------------- Hero --------------------
col1, col2 = st.columns([1, 2])
with col1:
    st_lottie(ai_anim, height=250)
with col2:
    type_text("📘 SlideSense AI Platform")
    st.markdown("### Smart Learning | Smart Vision | Smart AI")

st.divider()

# -------------------- PDF ANALYZER --------------------
if page == "📘 PDF Analyzer":
    st_lottie(upload_anim, height=180)
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if st.session_state.vector_db is None:
            with st.spinner("🧠 Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=80
                )
                chunks = splitter.split_text(text)

                try:
                    asyncio.get_running_loop()
                except:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("PDF Ready 🚀")
        q = st.text_input("Ask your question")

        if q:
            with st.spinner("🤖 AI Thinking..."):
                docs = st.session_state.vector_db.similarity_search(q, k=5)
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                history = ""
                for x, y in st.session_state.chat_history[-5:]:
                    history += f"Q:{x}\nA:{y}\n"

                prompt = ChatPromptTemplate.from_template("""
History:
{history}

Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

                chain = create_stuff_documents_chain(llm, prompt)
                res = chain.invoke({
                    "context": docs,
                    "question": q,
                    "history": history
                })

                st.session_state.chat_history.append((q, res))

        st.markdown("## 💬 AI Conversation")
        for q, a in st.session_state.chat_history:
            st.markdown(f"🧑 **You:** {q}")
            st.markdown(f"🤖 **AI:** {a}")
            st.divider()

# -------------------- IMAGE QUESTION ANSWERING --------------------
if page == "🖼 Image Q&A":
    st_lottie(upload_anim, height=180)
    img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if img_file:
        img = Image.open(img_file).convert("RGB")
        st.image(img, use_column_width=True)

        question = st.text_input("Ask a question about the image")

        if question:
            with st.spinner("🤖 Analyzing image..."):
                st_lottie(ai_anim, height=120)
                answer = answer_image_question(img, question)

            st.success(answer)
