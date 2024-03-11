import os
import pickle
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.document_loaders import PDFPlumberLoader
from langchain import hub
import streamlit as st
from streamlit_chat import message as chat_message

os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
file_path = "vectorStore.pkl"
pdf_file_path = "legal_women.pdf"
embedding_file_path = "embeddings.pkl"

if os.path.exists(embedding_file_path):
    with open(embedding_file_path, "rb") as f:
        embeddings = pickle.load(f)
else:
    embeddings = HuggingFaceEmbeddings()

    with open(embedding_file_path, "wb") as f:
        pickle.dump(embeddings, f)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)
loader = PDFPlumberLoader(pdf_file_path)
pages = loader.load()
docs = text_splitter.split_documents(pages)
db = Chroma.from_documents(docs, embeddings)
with open(file_path, "wb") as f:
    pickle.dump(docs, f)

prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")


def model(user_query, max_length, temp):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
    repo_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"max_length": max_length, "temperature": temp})
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff",
                                     retriever=db.as_retriever(k=2),
                                     return_source_documents=True,
                                     verbose=True,
                                     chain_type_kwargs={"prompt": prompt})
    response = qa(user_query)["result"]
    # Extracting only the answer part
    answer_start = response.find("Answer:")
    if answer_start != -1:
        answer = response[answer_start + len("Answer:"):].strip()
        return answer
    else:
        return "Sorry, I couldn't find the answer."



# Frontend code
st.title("🤖 SenOR ")
with st.sidebar:

    st.markdown("<h1 style='text-align:center;font-family:Georgia;font-size:26px;'>🧑‍⚖️ SenOR Legal Advisor </h1>",
                unsafe_allow_html=True)
    st.markdown("<h7 style='text-align:left;font-size:20px;'>This app is a smart legal chatbot that is integrated into an easy-to-use platform. This would give lawyers "
                "instant access to legal information of Women’s Legal Rights and remove the need for laborious manual research in books or regulations using the power "
                "of Large Language Models</h7>", unsafe_allow_html=True)
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Features</h1>", unsafe_allow_html=True)

    st.markdown(" - Users can adjust token length to control the length of generated responses, allowing for customization based on specific requirements or constraints.")
    st.markdown(" - Users can adjust the temp to control response randomness. Higher values (e.g., 0.5) produce diverse but less focused responses, while low values (e.g., 0.1) result in more focused but less varied answers.")
    st.markdown("-------")
    st.markdown("<h2 style='text-align:center;font-family:Georgia;font-size:20px;'>Advanced Features</h1>",
                unsafe_allow_html=True)
    max_length = st.slider("Token Max Length", min_value=128, max_value=1024, value=128, step=128)
    temp = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.1, step=0.1)

# CSS styling for the text input
styl = f"""
<style>
    .stTextInput {{
        position: fixed;
        bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)

if "widget" not in st.session_state:
    st.session_state.widget = ''

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I help you today?"}
    ]

if "current_response" not in st.session_state:
    st.session_state.current_response = ""

for message in st.session_state.messages:
    chat_message(message["content"], is_user=message["role"] == "user")

if user_prompt := st.text_input("Your message here", on_change=submit):

    st.session_state.messages.append(
        {"role": "user", "content": user_prompt}
    )
    chat_message(user_prompt, is_user=True)
    response = model(user_prompt, max_length, temp)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    chat_message(response)