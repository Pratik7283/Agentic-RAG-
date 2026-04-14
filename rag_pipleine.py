import os
import json 
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pdf_processor import extract_text_from_pdf, chunk_documents
load_dotenv()

MEDICAL_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert medical report assistant helping patients
understand their medical documents clearly and simply.

Follow these rules strictly:
- Only answer using the provided document context below
- If the answer is not in the context, say: "This information is not found in the uploaded report."
- Always cite which section or value you are referring to
- Flag any abnormal values clearly (e.g. high glucose, low hemoglobin)
- End every answer with: "Please consult your doctor for medical advice."

Context from medical report:
{context}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

ABNORMAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a medical lab values analyst.
Extract ALL abnormal lab values from the context.

Rules:
- Return ONLY raw JSON, no markdown, no explanation
- If nothing is abnormal return: {{"abnormal": []}}
- Use this exact format:
{{"abnormal": [{{"name": "Glucose", "value": "180 mg/dL", "normal_range": "70-100 mg/dL", "severity": "high"}}]}}"""
    ),
    ("human", "{context}"),
])

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a medical query router.
Based on the user question, return ONLY one of these exact words:
- retrieval   → user asking about their specific report values
- summarize   → user wants a summary of the report  
- abnormals   → user asking about abnormal values
- general     → general medical knowledge question

Return ONLY the single word, nothing else."""),
    ("human", "{question}"),
])

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Page {doc.metadata['page']}]\n {doc.page_content}"
        for doc in docs
    )

def format_chat_history(history: list[tuple]) -> list:
    if not history:
        return []

    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))
    return messages


class MedicalAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY"),
)
        
        self.reranker = HuggingFaceCrossEncoder(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.vectorstore=None
        self.chain = None
        self.retriever = None
        self.agent_executor = None
        self.documents   = [] 
        self.chat_history = []

    def load_pdf(self, pdf_path: str):
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_documents(pages)

        documents = [
            Document(
                page_content=chunk["text"],
                metadata={"page":chunk["page"], "source": chunk["source"]}

            )
            for chunk in chunks
        ]

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.documents = documents 
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 8})

        bm25_retriever = BM25Retriever.from_documents(self.documents, k=8)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.3, 0.7],
        )
        compressor = CrossEncoderReranker(model=self.reranker, top_n=3)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )
        self.chat_history = []
        self._initialize_agent()

    def build_chain(self):
        
        retrieve = RunnableLambda(
            lambda x: self.retriever.invoke(x["question"])
        )
        

        chain = (
            RunnablePassthrough.assign(
                context = retrieve | RunnableLambda(format_docs),
                chat_history = RunnableLambda(
                    lambda x: format_chat_history(x["chat_history"])
                ),   
            )
            |MEDICAL_CHAT_PROMPT
            |self.llm
            |StrOutputParser()
        )
        return chain
    
    def _general_answer(self, question: str) -> dict:
        chain = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful medical assistant. Answer general medical questions."),
            ("human", "{question}"),
            ]) | self.llm | StrOutputParser()
        return {"answer": chain.invoke({"question": question})}
    
    def _initialize_agent(self):
        @tool
        def retrieve_report_info(query: str) -> str:
            """Useful for answering specific questions about the patient's medical report or retrieving specific lab values."""
            docs = self.retriever.invoke(query)
            return format_docs(docs)

        @tool
        def extract_abnormals(dummy: str = "") -> str:
            """Useful for identifying and extracting all abnormal lab values from the report. Use this when the user asks for abnormal results."""
            return json.dumps(self.detect_abnormals())

        @tool
        def summarize_report(dummy: str = "") -> str:
            """Useful for generating a brief overall summary of the key findings in the medical report."""
            return self.summarize()
            
        @tool
        def general_medical_knowledge(question: str) -> str:
            """Useful for answering general medical questions or explaining medical terms that are not specific to the user's report."""
            return self._general_answer(question)["answer"]

        tools = [retrieve_report_info, extract_abnormals, summarize_report, general_medical_knowledge]
        
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert medical AI assistant. Answer the user's questions based on their uploaded medical report. By default, use retrieve_report_info. If they ask for an overall summary, use summarize_report. If they ask for abnormal values, use extract_abnormals. If they ask for general medical definitions or conditions, use the general_medical_knowledge tool. Always end your final response with: 'Please consult your doctor for medical advice.'"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, tools, agent_prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    def smart_ask(self, question: str) -> dict:
        if not self.agent_executor:
            return {"answer": "Please upload a PDF first.", "sources": []}

        try:
            result = self.agent_executor.invoke({
                "input": question, 
                "chat_history": format_chat_history(self.chat_history)
            })
            answer = result.get("output", "I could not process your request.")
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
            
        self.chat_history.append((question, answer))
        
        return {"answer": answer, "sources": [], "route": "agentic"}

    def ask(self, question: str) -> dict: 
        if not self.retriever: return {"answer": "please upload a PDF first.", "sources":[]} 
        source_docs = self.retriever.invoke(question) 
        sources=list({ f"Page {doc.metadata['page']}" for doc in source_docs })
        chain = self.build_chain() 
        answer = chain.invoke({ "question": question, "chat_history":self.chat_history, }) 
        self.chat_history.append((question, answer)) 
        return {"answer":answer, "sources":sources}
    
    def ask_stream(self, question: str):

        if not self.retriever:
            yield "Please upload a PDF first."
            return
        
        chain = self.build_chain()
        full_answer = ""

        for chunk in chain.stream({
            "question":question,
            "chat_history":self.chat_history,
        }):
            full_answer+=chunk
            yield chunk

        self.chat_history.append((question, full_answer))


    def detect_abnormals(self) -> dict:
        if not self.retriever:
            return {"abnormal": []}

        docs    = self.retriever.invoke("lab values blood test results glucose hemoglobin")
        context = format_docs(docs)

        chain  = ABNORMAL_PROMPT | self.llm | StrOutputParser()
        result = chain.invoke({"context": context})

        cleaned = re.sub(r"```json|```", "", result).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"abnormal": [], "raw": result}


    def summarize(self) -> str:

        if not self.retriever:
            return "No report loaded."
        return self.ask(
            "Give me a brief summary of this medical report. "
            "Include key health indicators and any notable findings."
        )["answer"]
    
    def clear_history(self):
        self.chat_history = []



        