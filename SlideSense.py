import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import uuid
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate


# ================= FIREBASE INIT =================
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

db = firestore.client()

st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")

USER_ID = "demo_user"   # replace with real auth id


# ================= SESSION STATE =================
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None


# ================= LLM =================
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash")


# ================= FIREBASE FUNCTIONS =================
def load_chats():
    chats = {}
    docs = db.collection("users").document(USER_ID).collection("chats").stream()
    for doc in docs:
        chats[doc.id] = doc.to_dict()
    return chats


def save_chat(chat_id):
    db.collection("users").document(USER_ID).collection("chats").document(chat_id).set(
        st.session_state.chats[chat_id]
    )


def delete_chat(chat_id):
    db.collection("users").document(USER_ID).collection("chats").document(chat_id).delete()
    st.session_state.chats.pop(chat_id, None)


def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": None,  # 🔥 start empty
        "messages": [],
        "pdf_id": None
    }
    st.session_state.current_chat_id = chat_id
    st.session_state.vector_store = None


# ================= INITIAL LOAD =================
if not st.session_state.chats:
    st.session_state.chats = load_chats()

if not st.session_state.current_chat_id:
    if st.session_state.chats:
        st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
    else:
        create_new_chat()


# ================= SIDEBAR =================
st.sidebar.title("💬 Chats")

if st.sidebar.button("➕ New Chat"):
    create_new_chat()
    st.rerun()

for cid, chat in st.session_state.chats.items():
    col1, col2 = st.sidebar.columns([4, 1])

    title = chat["title"] if chat["title"] else "New Chat"

    if col1.button(title, key=f"select_{cid}"):
        st.session_state.current_chat_id = cid
        st.session_state.vector_store = None   # reset vector when switching
        st.rerun()

    if col2.button("🗑", key=f"delete_{cid}"):
        delete_chat(cid)
        st.session_state.vector_store = None
        if st.session_state.chats:
            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
        else:
            create_new_chat()
        st.rerun()


# ================= MAIN =================
st.title("📘 SlideSense AI")

chat_id = st.session_state.current_chat_id
chat_data = st.session_state.chats[chat_id]


# ================= PDF UPLOAD =================
pdf = st.file_uploader("Upload PDF", type="pdf")

if pdf:
    pdf_id = f"{pdf.name}_{pdf.size}"

    # Only rebuild if new PDF for this chat
    if chat_data["pdf_id"] != pdf_id:

        with st.spinner("Processing PDF..."):

            reader = PdfReader(pdf)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=80
            )

            chunks = splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vector_store = FAISS.from_texts(chunks, embeddings)
            chat_data["pdf_id"] = pdf_id

            save_chat(chat_id)


# ================= CHAT INPUT =================
question = st.chat_input("Ask about this PDF")

if question:

    # Add user message
    chat_data["messages"].append({
        "role": "user",
        "content": question
    })

    # 🔥 TITLE AUTO GENERATION (only first user message)
    if chat_data["title"] is None:
        chat_data["title"] = question[:30] + (
            "..." if len(question) > 30 else ""
        )

    # Generate answer
    if st.session_state.vector_store is None:
        answer = "Please upload a PDF first."
    else:
        llm = load_llm()
        docs = st.session_state.vector_store.similarity_search(question, k=5)

        prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

Answer only from document.
""")

        chain = create_stuff_documents_chain(llm, prompt)
        result = chain.invoke({"context": docs, "question": question})
        answer = result if isinstance(result, str) else result.get("output_text", "")

    # Add assistant message
    chat_data["messages"].append({
        "role": "assistant",
        "content": answer
    })

    # Save AFTER updating title + messages
    save_chat(chat_id)

    st.rerun()


# ================= DISPLAY CHAT =================
for msg in chat_data["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
