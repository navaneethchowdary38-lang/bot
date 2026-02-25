import streamlit as st
from datetime import datetime
import uuid
from PyPDF2 import PdfReader
from PIL import Image
import io

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
from langchain_core.messages import HumanMessage


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
    "vector_db": None
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


# -------------------- FIRESTORE --------------------
def create_new_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    db.collection("users").document(user_id).collection("chats").document(chat_id).set({
        "mode": mode,
        "created_at": datetime.utcnow(),
        "title": "New Chat"
    })
    return chat_id


def save_message(user_id, chat_id, role, content):
    db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages").add({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        })


def load_user_chats(user_id, mode):
    chats = db.collection("users").document(user_id).collection("chats") \
        .where("mode", "==", mode) \
        .order_by("created_at", direction=firestore.Query.DESCENDING) \
        .stream()

    return [(doc.id, doc.to_dict()["title"]) for doc in chats]


def load_messages(user_id, chat_id):
    messages = db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages") \
        .order_by("timestamp").stream()

    return [(doc.to_dict()["role"], doc.to_dict()["content"]) for doc in messages]


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
            else:
                st.error("Signup failed")

    st.stop()


# -------------------- SIDEBAR --------------------
st.sidebar.success(f"👤 {st.session_state.email}")

if st.sidebar.button("🚪 Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

mode = st.sidebar.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"

st.sidebar.markdown("## 💬 Your Chats")

user_chats = load_user_chats(st.session_state.user_id, st.session_state.mode)

for chat_id, title in user_chats:
    col1, col2 = st.sidebar.columns([4, 1])
    icon = "📘" if st.session_state.mode == "PDF" else "🖼"

    if col1.button(f"{icon} {title}", key=f"open_{chat_id}"):
        st.session_state.current_chat_id = chat_id
        st.rerun()

    if col2.button("🗑", key=f"delete_{chat_id}"):
        db.collection("users").document(st.session_state.user_id) \
          .collection("chats").document(chat_id).delete()

        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None

        st.rerun()

if st.sidebar.button("➕ New Chat"):
    new_chat_id = create_new_chat(st.session_state.user_id, st.session_state.mode)
    st.session_state.current_chat_id = new_chat_id
    st.session_state.vector_db = None
    st.rerun()


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )


# -------------------- MAIN CONTENT --------------------
if not st.session_state.current_chat_id:

    st.markdown("## 👋 Welcome to SlideSense AI")
    st.markdown("### 🚀 AI Powered PDF & Image Analyzer")
    st.info("Select 'New Chat' from the sidebar to begin.")

else:

    if st.session_state.mode == "PDF":

        st.markdown("## 📘 PDF Analyzer")
        pdf = st.file_uploader("Upload PDF", type="pdf")

        if pdf and st.session_state.vector_db is None:
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

    else:

        st.markdown("## 🖼 Image Question Answering")
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        if img_file:
            image = Image.open(img_file).convert("RGB")
            st.image(image, use_column_width=True)

    # -------------------- LOAD MESSAGES --------------------
    messages = load_messages(
        st.session_state.user_id,
        st.session_state.current_chat_id
    )

    for role, content in messages:
        if role == "user":
            st.markdown(f"🧑 {content}")
        else:
            st.markdown(f"🤖 {content}")

    question = st.chat_input("Ask something...")

    if question:

        save_message(
            st.session_state.user_id,
            st.session_state.current_chat_id,
            "user",
            question
        )

        # -------- PDF ANSWER --------
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

                result = chain.invoke({
                    "context": docs,
                    "question": question
                })

                answer = result.get("output_text", "") if isinstance(result, dict) else result

        import base64

# -------- IMAGE ANSWER (Gemini Vision Only FIXED) --------
else:
    if not img_file:
        answer = "Please upload an image first."
    else:
        with st.spinner("🖼 Analyzing image..."):

            llm = load_llm()

            # Convert image to base64
            image_bytes = img_file.getvalue()
            encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            response = llm.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                },
                            },
                        ]
                    )
                ]
            )

            answer = response.content

        save_message(
            st.session_state.user_id,
            st.session_state.current_chat_id,
            "assistant",
            answer
        )

        st.rerun()
