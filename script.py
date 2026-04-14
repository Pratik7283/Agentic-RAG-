import streamlit as st
import requests
import time

# Configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Med-Agent AI", layout="wide")

st.title("🩺 Medical Agentic Assistant")
st.markdown("---")

with st.sidebar:
    st.header("1. Ingestion")
    uploaded_file = st.file_uploader("Upload Medical Report (PDF)", type="pdf")
    
    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Analyzing PDF..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_URL}/upload", files=files)
                if response.status_code == 200:
                    st.success("Document Indexed in FAISS!")
                else:
                    st.error("Upload Failed.")
        else:
            st.warning("Please upload a file first.")

    st.markdown("---")
    st.header("2. Quick Actions")
    if st.button("Clear Memory"):
        requests.post(f"{API_URL}/clear")
        st.session_state.messages = []
        st.rerun()

st.subheader("📋 Automated Lab Analysis")
if st.button("Extract Anomalies"):
    with st.spinner("Scanning for abnormal values..."):
        res = requests.get(f"{API_URL}/abnormals")
        if res.status_code == 200:
            data = res.json()
            # Displaying the Structured JSON output in a clean table
            st.table(data) 
        else:
            st.info("Upload a document to detect anomalies.")

st.markdown("---")

st.subheader("💬 Smart Medical Query")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about lab trends or summaries..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Agent Reasoning...", expanded=False) as status:
            st.write("Routing query to Medical Analysis tool...")
            # Calling your 'smart_ask' endpoint
            response = requests.post(f"{API_URL}/smart_ask", json={"question": prompt})
            status.update(label="Analysis Complete!", state="complete")
        
        if response.status_code == 200:
            ans = response.json()
            # Handle if answer is a dict or string based on your pipeline.ask output
            full_res = ans.get("output", str(ans)) 
            st.markdown(full_res)
            st.session_state.messages.append({"role": "assistant", "content": full_res})