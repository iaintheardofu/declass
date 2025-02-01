import os
import json
from datetime import datetime

import streamlit as st
import openai
import pandas as pd
import altair as alt

# LangChain imports
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Set up your OpenAI API key (ensure it's set in your environment)
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
                with open(os.path.join(policy_dir, filename), "r", encoding="utf-8") as f:
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
    The prompt instructs the LLM to act as a proactive agent.
    """
    prompt = f"""You are an expert report generator and proactive agent.
Using the source document and the report template below, generate a detailed, structured report.
Include a brief summary of next steps and actionable recommendations.

Source Document:
{document_text}

Report Template:
{template_text}

Produce a comprehensive report that is well-structured and actionable."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert report generator and proactive agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        report = response["choices"][0]["message"]["content"].strip()
        return report
    except Exception as e:
        return f"Error generating report: {str(e)}"

def feith_api_submit(document_id: str, review_report: dict):
    """
    Placeholder function to simulate integration with FEITH.
    """
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
st.write("A collaboration with NVIDIA – Transforming document analysis and report generation with agentic behavior.")

# Sidebar for policy vectorstore configuration (for declassification review)
st.sidebar.header("Policy Document Setup")
policy_dir = st.sidebar.text_input("Policy Documents Directory", "./policy_docs")
if st.sidebar.button("Load & Build Policy Vectorstore"):
    with st.spinner("Loading policy documents..."):
        policy_docs = load_policy_documents(policy_dir)
        if not policy_docs:
            st.sidebar.error("No policy documents found in the specified directory!")
        else:
            policy_vectorstore = build_vectorstore_from_texts(policy_docs)
            st.session_state.policy_vectorstore = policy_vectorstore
            st.sidebar.success(f"Loaded {len(policy_docs)} policy documents and built vectorstore.")

# Create three tabs: Declassification Review, Chat with Document, Report Generation
tab1, tab2, tab3 = st.tabs(["Declassification Review", "Chat with Document", "Report Generation"])

# --- Tab 1: Declassification Review ---
with tab1:
    st.header("Declassification Review")
    document_query = st.text_area("Enter document text for declassification review (or upload a document):", height=200)
    uploaded_file = st.file_uploader("Or Upload a Document", type=["txt", "pdf"], key="declass_upload")
    if uploaded_file is not None:
        try:
            document_text = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            document_text = ""
        st.text_area("Uploaded Document Content", value=document_text, height=200)
        document_query = document_text
    if st.button("Run Declassification Analysis"):
        if "policy_vectorstore" not in st.session_state:
            st.error("Please build the policy vectorstore first via the sidebar!")
        elif not document_query:
            st.error("Please provide document text or upload a document!")
        else:
            query_prompt = (
                "You are an expert document analyst and proactive agent. "
                "Review the following document against established declassification policies "
                "and provide a detailed analysis, citing relevant policy sections and recommending next steps for declassification if applicable.\n\n"
                f"Document:\n{document_query}\n"
            )
            with st.spinner("Analyzing document..."):
                review_report = query_vectorstore(st.session_state.policy_vectorstore, query_prompt)
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

# --- Tab 2: Chat with Document ---
with tab2:
    st.header("Chat with Document")
    uploaded_chat_file = st.file_uploader("Upload a Document for Chat", type=["txt", "pdf"], key="chat_upload")
    if uploaded_chat_file is not None:
        try:
            file_text = uploaded_chat_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            file_text = ""
        st.text_area("Uploaded Document Content", value=file_text, height=200)
        if file_text:
            chat_vectorstore = build_vectorstore_from_texts([file_text])
            st.session_state.chat_vectorstore = chat_vectorstore
            st.success("Document vectorstore built for chat!")
    if "chat_vectorstore" in st.session_state:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        st.subheader("Chat with Document")
        chat_input = st.text_input("Enter your query for the document:")
        if st.button("Send Query"):
            if chat_input:
                response = query_vectorstore(st.session_state.chat_vectorstore, chat_input)
                st.session_state.chat_history.append({"query": chat_input, "response": response})
        if st.session_state.get("chat_history"):
            for msg in st.session_state.chat_history:
                st.markdown(f"**User:** {msg['query']}")
                st.markdown(f"**Response:** {msg['response']}")

# --- Tab 3: Report Generation ---
with tab3:
    st.header("Report Generation")
    st.write("Upload your source document and provide a report template to generate a structured report.")
    source_file = st.file_uploader("Upload Source Document", type=["txt", "pdf"], key="report_source")
    template_text = st.text_area("Enter Report Template", height=150, placeholder="e.g., Markdown template with section instructions...")
    if source_file is not None:
        try:
            source_text = source_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading source file: {e}")
            source_text = ""
        st.text_area("Source Document Content", value=source_text, height=200)
    if st.button("Generate Report"):
        if not source_file or not template_text:
            st.error("Please provide both a source document and a report template.")
        else:
            with st.spinner("Generating report..."):
                report = generate_report(source_text, template_text)
            st.subheader("Generated Report")
            st.write(report)

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
st.markdown("© Document Research Assistant")
