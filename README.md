# 🩺 Agentic Medical Report Analyser
An AI-powered diagnostic assistant that uses **Agentic RAG** to interpret medical reports with clinical precision.

---

## 🚀 The Core Innovation: "Agentic" vs "Standard" RAG
Unlike traditional RAG systems that simply retrieve and summarize, this project uses a **ReAct (Reasoning and Acting)** Agent.

* **Standard RAG:** User Query → Search → Answer.
* **Agentic RAG (This Project):** User Query → **Thought Process** → **Tool Selection** (Hybrid Search vs. Anomaly Detector) → **Observation** → **Final Clinical Answer**.

---

## 🛠️ Tech Stack
* **LLM:** Llama 3.1 (via Groq for ultra-fast inference)
* **Framework:** LangChain (ReAct Agent logic)
* **Backend:** FastAPI
* **Database:** PostgreSQL (Metadata storage)
* **Vector DB:** FAISS (High-speed semantic retrieval)
* **Frontend:** Streamlit

---

## 🔥 Key Features
* **Hybrid Retrieval:** Combines **BM25 Keyword Search** (to catch specific medical acronyms like MCV, RDW) with **FAISS Semantic Search** (to understand concepts like "high blood sugar" vs "hyperglycemia").
* **Medical Anomaly Detection:** A specialized tool that flags values outside of standard clinical reference ranges.
* **Smart Summarization:** Generates patient-friendly summaries of complex pathological data.
* **Traceable Reasoning:** The UI displays the Agent's "Thought" process, allowing users to see exactly how the AI arrived at its conclusion.

---

## 🏗️ System Architecture
1.  **Document Ingestion:** PDF reports are processed, chunked, and stored in a hybrid vector-keyword index.
2.  **The Orchestrator:** A LangChain agent acts as the "Brain," routing queries to the most relevant tool.
3.  **The Toolset:**
    * `search_tool`: Performs RAG to find specific facts.
    * `abnormal_values_tool`: Scans tables for flags (High/Low).
    * `summary_tool`: Condenses the entire report.
      
