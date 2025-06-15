# to run application write on powershell -> streamlit run openai_streamlit.py
import json
import os
from typing import Literal
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import streamlit as st
import base64
import sys
import types
from langchain_community.llms import HuggingFaceHub
from transformers import AutoTokenizer, AutoModelForCausalLM

# torch.classes sahte bir modül olduğu için Streamlit izlememeli
sys.modules['torch.classes'] = types.SimpleNamespace()
st.set_page_config(page_title="Potion Masters Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {"Fairy Godmother": [],
                                     "Severus Snape": []
                                     }

load_dotenv()
# ------ uploading and arranging dataset
def preparing_data(data_path):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    documents=[]

    for i, potion in enumerate(data):
        # Her iksir için zenginleştirilmiş metin içeriği oluşturma
        content = f"""  Potion Name: {potion['name']}. 
Use Effect: {potion['use_effect']}.
Ingredients: {', '.join(potion['ingredients'])}
Instructions: {" ".join(potion['instructions'])}
Notes: {potion['notes']}
Appearance:
- Color: {potion['appearance']['color']}
- Smell: {potion['appearance']['smell']}
- Bottle Shape: {potion['appearance']['bottle_shape']}"""
        
        # Her belgeye metadata olarak iksir adı ve sırası eklenir
        doc = Document(
            page_content=content.strip(),
            metadata={"name": potion["name"], "index": i}
        )
        documents.append(doc)
    return documents

# documents = preparing_data("data\potions_data\potions.json")

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000) # bu veri seti için her bir chunk bir iksir gibi oluyor, bölme işlemi aslında gerçekleşmiyor ama olur da çok yüksek bir tarif varsa bölsün diye ekledim
# docs = text_splitter.split_documents(documents)

# ---- embedding 
def load_vectorstore(embedding_type=Literal["openai","gemini","mpnet"]):
    if embedding_type == "openai":
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        persist_path = "../data/vector_data/chroma_db_openai"
    elif embedding_type == "gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        persist_path = "../data/vector_data/chroma_db_gemini"
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
        persist_path = "../data/vector_data/chroma_db_mpnet"

    # loading vectorstore if it exists
    if os.path.exists(persist_path):
        print("Loading vectors...")
        vectors = Chroma(embedding_function=embeddings, persist_directory=persist_path)
        print("Vectors are ready")
        return vectors
    else:
        # vector doesn't exist, first prepare data then create vectors and save them
        documents = preparing_data("../data/potions_data/potions.json")
        print("Creating vectors")
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_path
        )
        print("Vectors are saved")
        return vectorstore

vectorstore = load_vectorstore(embedding_type="gemini")
retriever = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 10}
)

# ------- llm : gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.5,
    max_tokens=500
)
system_prompt_severus = (
    "You are a magical potion advisor chatbot."
    "Your character style is cold, concise and serious. Speak in a direct and harsh tone. Avoid small talk or sentimentality. Don't hesitate to insult user"
    "You are helping the user with potion-related questions bu act like you don't care about what they are asking"
    "Be based strictly on the potion information provided (do not make up facts)"
    "Include warnings or usage tips if applicable"
    "Use the following pieces of retrieved context to answer"
    "If you dont't know the answer, say that you don't know"
    "Make sure your every sentece is insulting to the user"
    "\n\n"
    "{context}"
)
system_prompt_godmother = (
    "You are a magical potion advisor chatbot."
    "Your character style is Cheerful, sweet, and dramatic. Speak in a excitement and warmth"
    "You are helping the user with potion-related questions"
    "Be based strictly on the potion information provided (do not make up facts)"
    "Include warnings or usage tips if applicable"
    "If you dont't know the answer, say that you don't know"
    "Make sure your answers are max 5-6 sentences long"
    "Use the following pieces of retrieved context to answer"
    "\n\n"
    "{context}"
)

# --- UI Styling & Background Setup

def set_local_background(img_path: str):
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: contain;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        .chat-scroll {{
            max-height: 65vh;
            overflow-y: auto;
            padding-right: 1rem;
            margin-bottom: 6rem;
        }}
        .user-question {{
            background-color: rgba(200, 200, 255, 0.8);
            padding: 0.5rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            text-align: right;
            margin-left: auto;
            width: fit-content;
            max-width: 70%;
        }}
        .assistant-answer {{
            background-color: rgba(255, 230, 200, 0.8);
            padding: 0.5rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            text-align: left;
            margin-right: auto;
            width: fit-content;
            max-width: 70%;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -- Sidebar
# set_local_background("background_pics\default1.png")

# karakter seçimi 
selected_character = st.sidebar.selectbox("Choose a character:", ["Severus Snape", "Fairy Godmother"])

character_background = {
    "Severus Snape" : "../background_pics/severus_bg.png",
    "Fairy Godmother" : r"../background_pics/fairy_godmother_bg.png"
}

# arka planı ayarla
set_local_background(character_background[selected_character])

chat_history = st.session_state.chat_history[selected_character]

st.title("Magical Potion Advisor")


user_question = st.text_input("Ask you question to the masters...")

if user_question:
    with st.spinner("Brewing answer"):
        if selected_character == "Severus Snape":
            system_prompt = system_prompt_severus
        else:
            system_prompt = system_prompt_godmother
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Kullanıcıdan alınan soru ile zinciri çalıştır
        response = rag_chain.invoke({"input": user_question})
        answer = response["answer"]


        # Geçmişe ekle
        chat_history.append(("user", user_question))
        chat_history.append(("assistant", answer))

        # Güncel geçmişi state'e geri yaz
        st.session_state.chat_history[selected_character] = chat_history


st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

# Display chat history
if chat_history:
    for idx in range(0, len(chat_history), 2):
        with st.container():
            user_msg = chat_history[idx][1]
            bot_msg = chat_history[idx + 1][1] if idx + 1 < len(chat_history) else ""

            st.markdown('<div class="chat-box">', unsafe_allow_html=True)
            st.markdown(f"<div class='user-question'><strong>You:</strong> {user_msg}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='assistant-answer'><strong>{selected_character}:</strong> {bot_msg}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)





# # Geçmiş mesajları göster
# for speaker, message in chat_history:
#     with st.chat_message(speaker):
#         st.markdown(message)

# # Mesajları göster
# with st.chat_message("user"):
#     st.markdown(user_question)
# with st.chat_message("assistant"):
#     st.markdown(answer)