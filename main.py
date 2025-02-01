import os
import streamlit as st
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, Document
from datetime import datetime
import json
import openai

# Set up your OpenAI API key (ensure it's set in your environment)
openai.api_key = os.getenv("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")

# --- Utility Functions ---

def load_policy_documents(policy_dir: str) -> list:
    """
    Load policy and guidance documents from a specified directory.
    This should include EO 13526, DoDM 5200.01, NARA documents, SCGs, ISOO guidance, etc.
    """
    docs = SimpleDirectoryReader(policy_dir).load_data()
    return docs

def load_example_documents(example_dir: str) -> list:
    """
    Load example declassified documents (e.g., JFK files) for the POC.
    """
    docs = SimpleDirectoryReader(example_dir).load_data()
    return docs

def build_index(documents: list) -> GPTSimpleVectorIndex:
    """
    Build a LlamaIndex vector index from a list of Document objects.
    """
    index = GPTSimpleVectorIndex(documents)
    return index

def query_declassification(index: GPTSimpleVectorIndex, query: str) -> str:
    """
    Query the index with a prompt, retrieve relevant paragraphs, and have the LLM provide a recommendation.
    """
    response = index.query(query, response_mode="compact")
    return str(response)

def feith_api_submit(document_id: str, review_report: dict):
    """
    Placeholder function to simulate integration with FEITH.
    In production, this function would perform an API POST/PUT to update the document status.
    """
    st.info(f"Simulated FEITH API update for document {document_id}.")
    with open("feith_integration_log.json", "a") as log_file:
        log_file.write(json.dumps({
            "document_id": document_id,
            "review_report": review_report,
            "timestamp": datetime.now().isoformat()
        }) + "\n")

# --- Streamlit App ---

st.set_page_config(page_title="Leyden Declassification Review Hub", layout="wide")

st.title("Leyden Solutions: Declassification Review Hub")
st.write("A Proof-of-Concept for automated document review using LLM & Retrieval Augmented Generation.")

# Sidebar for configuration and file uploads
st.sidebar.header("Configuration & Uploads")
policy_dir = st.sidebar.text_input("Policy Documents Directory", "./policy_docs")
example_dir = st.sidebar.text_input("Example Documents Directory (e.g., JFK files)", "./example_docs")

if st.sidebar.button("Load & Build Index"):
    with st.spinner("Loading documents..."):
        policy_docs = load_policy_documents(policy_dir)
        st.sidebar.success(f"Loaded {len(policy_docs)} policy documents.")
        policy_index = build_index(policy_docs)
    st.session_state.policy_index = policy_index
    st.success("Policy Index Built Successfully!")

# Main chat interface section
st.header("Document Review Chat Interface")
document_query = st.text_area("Enter document text (or upload a document below)", height=200)

uploaded_file = st.file_uploader("Or Upload a Document", type=["txt", "pdf"])
if uploaded_file is not None:
    document_text = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded Document Content", value=document_text, height=200)
    document_query = document_text

if st.button("Run Declassification Analysis"):
    if "policy_index" not in st.session_state:
        st.error("Please build the policy index first via the sidebar!")
    elif not document_query:
        st.error("Please provide document text or upload a document!")
    else:
        prompt = (
            "Given the following document, please review it against the declassification policies (EO 13526, DoDM 5200.01, NARA, "
            "SCGs, ISOO guidance, and publicly available information). Identify and cite the specific paragraph(s) from the policies "
            "that indicate whether the document should or should not be declassified. Then, recommend the next steps to achieve declassification if applicable.\n\n"
            f"Document:\n{document_query}\n"
        )
        with st.spinner("Analyzing document..."):
            review_report = query_declassification(st.session_state.policy_index, prompt)
        st.subheader("Declassification Review Report")
        st.write(review_report)
        document_id = "example_document_001"
        review_data = {
            "document_id": document_id,
            "review_report": review_report,
            "reviewed_at": datetime.now().isoformat()
        }
        feith_api_submit(document_id, review_data)
        st.success("Review complete and FEITH update simulated!")

# --- Analytics Dashboard Section ---

st.header("Real-Time Analytics Dashboard")
analytics_data = {
    "Documents Reviewed": 25,
    "Declassification Recommended": 10,
    "Maintained Classification": 15,
    "Avg. Processing Time (sec)": 45
}
col1, col2, col3, col4 = st.columns(4)
col1.metric("Documents Reviewed", analytics_data["Documents Reviewed"])
col2.metric("Declassification Recommended", analytics_data["Declassification Recommended"])
col3.metric("Maintained Classification", analytics_data["Maintained Classification"])
col4.metric("Avg. Processing Time (sec)", analytics_data["Avg. Processing Time (sec)"])
st.markdown("---")
st.write("Detailed analytics and logs would be displayed here. In a production system, these would update in real-time as documents are processed.")

import pandas as pd
import altair as alt

df = pd.DataFrame({
    "Outcome": ["Declassified", "Not Declassified"],
    "Count": [analytics_data["Declassification Recommended"], analytics_data["Maintained Classification"]]
})
chart = alt.Chart(df).mark_bar().encode(
    x="Outcome",
    y="Count",
    color="Outcome"
).properties(
    width=600,
    height=300,
    title="Document Declassification Outcomes"
)
st.altair_chart(chart)
st.markdown("© Leyden Solutions")
