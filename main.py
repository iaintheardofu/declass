import os
import json
from datetime import datetime

import streamlit as st
import openai
import pandas as pd
import altair as alt

# LangChain imports (Updated for langchain-community)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Utility Functions ---

def load_policy_documents(policy_dir: str) -> list:
    """
    Load all .txt policy documents from the specified directory.
    Returns a list of document texts.
    """
    docs = []
    if os.path.isdir(policy_dir):
        for filename in os.listdir(policy_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(policy_dir, filename), "r", encoding="utf-8", errors="replace") as f:
                    docs.append(f.read())
    return docs

def build_vectorstore_from_texts(texts: list) -> FAISS:
    """
    Split the texts into chunks and build a FAISS vectorstore.
    """
    splitter = CharacterTextSplitter(separator=" ", chunk_size=500, chunk_overlap=50)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def query_vectorstore(vectorstore: FAISS, query: str) -> str:
    """
    Create a RetrievalQA chain to query the vectorstore.
    """
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    result = qa.run(query)
    return result

def generate_report(document_text: str, template_text: str) -> str:
    """
    Generate a structured report based on the source document and the provided template.
    """
    prompt = f"""You are an expert report generator.
Using the source document and the report template below, generate a structured, actionable report.

Source Document:
{document_text}

Report Template:
{template_text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert report generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error generating report: {str(e)}"

def feith_api_submit(document_id: str, review_report: dict):
    """Simulate integration with FEITH API."""
    st.info(f"Simulated FEITH API update for document {document_id}.")
    with open("feith_integration_log.json", "a") as log_file:
        log_file.write(json.dumps({
            "document_id": document_id,
            "review_report": review_report,
            "timestamp": datetime.now().isoformat()
        }) + "\n")

# --- Streamlit App ---

st.set_page_config(page_title="Document Research Assistant", layout="wide")
st.title("Document Research Assistant")

# Sidebar for policy document setup
st.sidebar.header("Policy Document Setup")
policy_dir = st.sidebar.text_input("Policy Documents Directory", "./policy_docs")
if st.sidebar.button("Load & Build Policy Vectorstore"):
    with st.spinner("Loading policy documents..."):
        policy_docs = load_policy_documents(policy_dir)
        if not policy_docs:
            st.sidebar.error("No policy documents found!")
        else:
            st.session_state.policy_vectorstore = build_vectorstore_from_texts(policy_docs)
            st.sidebar.success(f"Loaded {len(policy_docs)} documents and built vectorstore.")

# Tabs: Declassification Review, Chat with Document, Report Generation
tab1, tab2, tab3 = st.tabs(["Declassification Review", "Chat with Document", "Report Generation"])

# --- Tab 1: Declassification Review ---
with tab1:
    st.header("Declassification Review")
    document_query = st.text_area("Enter document text or upload a file:", height=200)
    uploaded_file = st.file_uploader("Upload Document", type=["txt", "pdf"])
    
    if uploaded_file:
        try:
            document_text = uploaded_file.read().decode("utf-8", errors="replace")
            st.text_area("Uploaded Content", value=document_text, height=200)
            document_query = document_text
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if st.button("Run Analysis"):
        if "policy_vectorstore" not in st.session_state:
            st.error("Please build the policy vectorstore first!")
        elif not document_query:
            st.error("Provide document text or upload a file!")
        else:
            with st.spinner("Analyzing..."):
                review_report = query_vectorstore(st.session_state.policy_vectorstore, document_query)
            st.subheader("Review Report")
            st.write(review_report)
            feith_api_submit("doc_001", {"report": review_report})

# --- Tab 2: Chat with Document ---
with tab2:
    st.header("Chat with Document")
    uploaded_chat_file = st.file_uploader("Upload Document", type=["txt", "pdf"])
    
    if uploaded_chat_file:
        try:
            file_text = uploaded_chat_file.read().decode("utf-8", errors="replace")
            st.text_area("Uploaded Content", value=file_text, height=200)
            st.session_state.chat_vectorstore = build_vectorstore_from_texts([file_text])
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if "chat_vectorstore" in st.session_state:
        chat_input = st.text_input("Enter query:")
        if st.button("Send Query"):
            if chat_input:
                response = query_vectorstore(st.session_state.chat_vectorstore, chat_input)
                st.write("**Response:**", response)

# --- Tab 3: Report Generation ---
with tab3:
    st.header("Report Generation")
    source_file = st.file_uploader("Upload Source Document", type=["txt", "pdf"])
    template_text = st.text_area("Enter Report Template", height=150)
    
    if source_file:
        try:
            source_text = source_file.read().decode("utf-8", errors="replace")
            st.text_area("Source Document", value=source_text, height=200)
        except Exception as e:
            st.error(f"Error reading source file: {e}")

    if st.button("Generate Report"):
        if not source_file or not template_text:
            st.error("Provide both a document and a template.")
        else:
            with st.spinner("Generating report..."):
                report = generate_report(source_text, template_text)
            st.subheader("Generated Report")
            st.write(report)

# --- Real-Time Analytics ---
st.header("Analytics Dashboard")
df = pd.DataFrame({"Outcome": ["Declassified", "Not Declassified"], "Count": [10, 15]})
chart = alt.Chart(df).mark_bar().encode(x="Outcome", y="Count", color="Outcome")
st.altair_chart(chart)
st.markdown("© Document Research Assistant")
