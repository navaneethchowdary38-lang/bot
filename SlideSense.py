import streamlit as st
from datetime import datetime
import uuid
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
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

db = firestore.client()


# -------------------- SESSION INIT --------------------
defaults = {
    "authenticated": False,
    "user_id": None,
    "email": None,
    "mode": "PDF",
    "current_chat_id": None,
    "vector_db": None,
    "edit_chat_id": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------- FIREBASE AUTH --------------------
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


# -------------------- FIRESTORE CHAT STRUCTURE --------------------
def create_new_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    db.collection("users").document(user_id) \
      .collection("chats").document(chat_id) \
      .set({
          "mode": mode,
          "created_at": datetime.utcnow(),
          "title": "New Chat"
      })
    return chat_id


def save_message(user_id, chat_id, role, content):
    messages_ref = db.collection("users") \
      .document(user_id) \
      .collection("chats") \
      .document(chat_id) \
      .collection("messages")

    messages_ref.add({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })

    # Auto-title on first user message
    if role == "user":
        msgs = list(messages_ref.stream())
        if len(msgs) == 1:
            short_title = content[:45] + "..." if len(content) > 45 else content
            db.collection("users") \
              .document(user_id) \
              .collection("chats") \
              .document(chat_id) \
              .update({"title": short_title})


def load_messages(user_id, chat_id):
    messages = db.collection("users") \
                 .document(user_id) \
                 .collection("chats") \
                 .document(chat_id) \
                 .collection("messages") \
                 .order_by("timestamp") \
                 .stream()

    return [(doc.to_dict()["role"], doc.to_dict()["content"]) for doc in messages]


# -------------------- HELPERS --------------------
def update_chat_title(user_id, chat_id, title):
    db.collection("users").document(user_id) \
      .collection("chats").document(chat_id) \
      .update({"title": title})


def delete_chat(user_id, chat_id):
    chat_ref = db.collection("users") \
        .document(user_id) \
        .collection("chats") \
        .document(chat_id)

    for msg in chat_ref.collection("messages").stream():
        msg.reference.delete()

    chat_ref.delete()


def group_chats_by_date(chats):
    today = datetime.utcnow().date()
    yesterday = today.fromordinal(today.toordinal() - 1)

    groups = {
        "Today": [],
        "Yesterday": [],
        "Last 7 Days": [],
        "Older": []
    }

    for chat_id, data in chats:
        created = data["created_at"].date()
        if created == today:
            groups["Today"].append((chat_id, data))
        elif created == yesterday:
            groups["Yesterday"].append((chat_id, data))
        elif (today - created).days <= 7:
            groups["Last 7 Days"].append((chat_id, data))
        else:
            groups["Older"].append((chat_id, data))

    return groups


# -------------------- CSS --------------------
st.markdown("""
<style>
.chat-user {
    background-color: #343541;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
    color: white;
}
.chat-ai {
    background-color: #444654;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 15px;
    color: white;
}

/* Sidebar UI */
.chat-item {
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
    background: rgba(255,255,255,0.05);
    transition: all 0.25s ease;
    cursor: pointer;
    animation: slideIn 0.3s ease;
}
.chat-item:hover {
    background: rgba(0, 255, 255, 0.12);
    transform: translateX(6px) scale(1.01);
    box-shadow: 0 0 12px rgba(0,255,255,0.2);
}
.chat-active {
    background: linear-gradient(90deg, #00f5ff33, #00ff9c22);
    border-left: 3px solid #00f5ff;
}
.chat-title {
    font-size: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.group-title {
    font-size: 12px;
    margin: 12px 0 6px 4px;
    opacity: 0.6;
    text-transform: uppercase;
}
@keyframes slideIn {
    from {opacity:0; transform:translateX(-10px);}
    to {opacity:1; transform:translateX(0);}
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOGIN --------------------
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
            else:
                st.error("Signup failed")

    st.stop()


# -------------------- SIDEBAR --------------------
st.sidebar.success(f"Logged in as {st.session_state.email}")

if st.sidebar.button("Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"

st.sidebar.markdown("## 💬 Your Chats")

chat_docs = db.collection("users") \
    .document(st.session_state.user_id) \
    .collection("chats") \
    .where("mode", "==", st.session_state.mode) \
    .order_by("created_at", direction=firestore.Query.DESCENDING) \
    .stream()

chats = [(doc.id, doc.to_dict()) for doc in chat_docs]
grouped = group_chats_by_date(chats)

for group, items in grouped.items():
    if not items:
        continue
    st.sidebar.markdown(f'<div class="group-title">{group}</div>', unsafe_allow_html=True)

    for chat_id, data in items:
        title = data.get("title", "New Chat")
        active = "chat-active" if st.session_state.current_chat_id == chat_id else ""

        st.sidebar.markdown(
            f'<div class="chat-item {active}"><span class="chat-title">{title}</span></div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.sidebar.columns([6,1,1])

        if c1.button(" ", key=f"open_{chat_id}"):
            st.session_state.current_chat_id = chat_id
            st.rerun()

        if c2.button("✏️", key=f"edit_{chat_id}"):
            st.session_state.edit_chat_id = chat_id

        if c3.button("🗑", key=f"del_{chat_id}"):
            delete_chat(st.session_state.user_id, chat_id)
            if st.session_state.current_chat_id == chat_id:
                st.session_state.current_chat_id = None
            st.rerun()

if st.session_state.edit_chat_id:
    st.sidebar.markdown("### ✏️ Rename Chat")
    new_title = st.sidebar.text_input("New title")
    if st.sidebar.button("Save Title"):
        update_chat_title(st.session_state.user_id, st.session_state.edit_chat_id, new_title)
        st.session_state.edit_chat_id = None
        st.rerun()

if st.sidebar.button("➕ New Chat"):
    new_chat_id = create_new_chat(st.session_state.user_id, st.session_state.mode)
    st.session_state.current_chat_id = new_chat_id
    st.session_state.vector_db = None
    st.rerun()


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


@st.cache_resource
def load_blip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    return processor, model, device


# -------------------- MAIN --------------------
if st.session_state.current_chat_id:

    if st.session_state.mode == "PDF":
        st.title("📘 PDF Analyzer")
        pdf = st.file_uploader("Upload PDF", type="pdf")

        if pdf and st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"

                splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
                chunks = splitter.split_text(text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

    if st.session_state.mode == "IMAGE":
        st.title("🖼 Image Q&A")
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    messages = load_messages(st.session_state.user_id, st.session_state.current_chat_id)

    for role, content in messages:
        if role == "user":
            st.markdown(f'<div class="chat-user">🧑 {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {content}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask something...")

    if question:
        save_message(st.session_state.user_id, st.session_state.current_chat_id, "user", question)

        if st.session_state.mode == "PDF":
            if st.session_state.vector_db is None:
                answer = "Please upload a PDF first."
            else:
                docs = st.session_state.vector_db.similarity_search(question, k=6)
                llm = load_llm()
                prompt = ChatPromptTemplate.from_template("""
Context:
{context}

Question:
{question}

If not found say:
Information not found in document.
""")
                chain = create_stuff_documents_chain(llm, prompt)
                result = chain.invoke({"context": docs, "question": question})
                answer = result.get("output_text", "") if isinstance(result, dict) else result

        else:
            if not img_file:
                answer = "Please upload an image first."
            else:
                processor, model, device = load_blip()
                img = Image.open(img_file).convert("RGB")
                inputs = processor(img, question, return_tensors="pt").to(device)
                outputs = model.generate(**inputs, max_length=30)
                answer = processor.decode(outputs[0], skip_special_tokens=True)

        save_message(st.session_state.user_id, st.session_state.current_chat_id, "assistant", answer)
        st.rerun()

else:
    st.title("🚀 Start a New Chat from Sidebar")
