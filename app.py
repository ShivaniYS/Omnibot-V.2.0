import streamlit as st
import os
import tempfile
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import base64

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# RAG Components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="OmniBot AI",
    page_icon="🤖",
    layout="wide"
)

# Clean CSS
st.markdown("""
<style>
    /* Clean up Streamlit defaults */
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #1f1f1f;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        font-weight: 500;
    }
    /* Simple message styles */
    .user-message {
        background-color: #f0f7ff;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0;
        border-left: 4px solid #0066cc;
    }
    .ai-message {
        background-color: #f9f9f9;
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0;
        border-left: 4px solid #00aa6c;
    }
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    defaults = {
        'conversation': [],
        'documents': {},
        'vectorstore': None,
        'processed_files': [],
        'api_keys': {
            'GROQ_API_KEY': st.secrets.get("GROQ_API_KEY", ""),
            'OPENAI_API_KEY': st.secrets.get("OPENAI_API_KEY", "")
        },
        'model': 'groq-llama-70b',
        'temperature': 0.7,
        'max_tokens': 1024
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# MODEL REGISTRY
# ============================================================================
class ModelManager:
    @staticmethod
    def get_model():
        model_id = st.session_state.model
        
        if model_id.startswith('groq'):
            api_key = st.session_state.api_keys.get('GROQ_API_KEY')
            if not api_key:
                return None
            model_name = 'llama-3.3-70b-versatile' if '70b' in model_id else 'llama-3.1-8b-instant'
            return ChatGroq(
                groq_api_key=api_key,
                model_name=model_name,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens
            )
        else:
            api_key = st.session_state.api_keys.get('OPENAI_API_KEY')
            if not api_key:
                return None
            model_name = 'gpt-4-turbo-preview' if 'gpt4' in model_id else 'gpt-3.5-turbo'
            return ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens
            )
    
    @staticmethod
    def get_available():
        available = []
        if st.session_state.api_keys.get('GROQ_API_KEY'):
            available.extend(['groq-llama-70b', 'groq-llama-8b'])
        if st.session_state.api_keys.get('OPENAI_API_KEY'):
            available.extend(['openai-gpt4', 'openai-gpt35'])
        return available

# ============================================================================
# RAG SYSTEM
# ============================================================================
class RAGSystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
    
    def process_files(self, uploaded_files):
        documents = []
        temp_files = []
        
        try:
            for uploaded_file in uploaded_files:
                ext = uploaded_file.name.split('.')[-1].lower()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
                    f.write(uploaded_file.getbuffer())
                    temp_path = f.name
                    temp_files.append(temp_path)
                
                # Load based on file type
                if ext == 'pdf':
                    loader = PyPDFLoader(temp_path)
                elif ext == 'txt':
                    loader = TextLoader(temp_path)
                elif ext == 'csv':
                    loader = CSVLoader(temp_path)
                else:
                    continue
                
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                documents.extend(docs)
            
            if documents:
                splits = self.splitter.split_documents(documents)
                vectorstore = FAISS.from_documents(splits, self.embeddings)
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = [f.name for f in uploaded_files]
                return True
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            for f in temp_files:
                try:
                    os.unlink(f)
                except:
                    pass
        return False
    
    def search(self, query):
        if not st.session_state.vectorstore:
            return []
        return st.session_state.vectorstore.similarity_search(query, k=3)
    
    def answer_with_context(self, query, context):
        llm = ModelManager.get_model()
        if not llm:
            return "Please configure API key"
        
        context_text = "\n".join([f"[Source {i+1}]: {doc.page_content[:400]}" 
                                 for i, doc in enumerate(context)])
        
        prompt = f"""Answer based on this context:

{context_text}

Question: {query}

If the answer isn't in the context, say so clearly."""
        
        try:
            chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
            return chain.invoke({})
        except Exception as e:
            return f"Error: {str(e)}"

# ============================================================================
# SIDEBAR
# ============================================================================
def show_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Keys
        st.subheader("API Keys")
        groq_key = st.text_input("Groq Key", 
                                value=st.session_state.api_keys['GROQ_API_KEY'],
                                type="password")
        openai_key = st.text_input("OpenAI Key", 
                                  value=st.session_state.api_keys['OPENAI_API_KEY'],
                                  type="password")
        
        if groq_key != st.session_state.api_keys['GROQ_API_KEY']:
            st.session_state.api_keys['GROQ_API_KEY'] = groq_key
        
        if openai_key != st.session_state.api_keys['OPENAI_API_KEY']:
            st.session_state.api_keys['OPENAI_API_KEY'] = openai_key
        
        # Model Selection
        st.subheader("Model")
        available = ModelManager.get_available()
        if available:
            model_names = {
                'groq-llama-70b': 'Llama 70B (Fast)',
                'groq-llama-8b': 'Llama 8B (Very Fast)',
                'openai-gpt4': 'GPT-4 (Best)',
                'openai-gpt35': 'GPT-3.5 (Cheap)'
            }
            selected = st.selectbox(
                "Choose model",
                available,
                format_func=lambda x: model_names.get(x, x),
                index=0
            )
            st.session_state.model = selected
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            with col2:
                st.session_state.max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 256)
        else:
            st.warning("Add API keys first")
        
        # Stats
        st.subheader("Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chats", len(st.session_state.conversation))
        with col2:
            st.metric("Files", len(st.session_state.processed_files))
        
        # Clear buttons
        st.subheader("Manage")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()
        
        if st.button("Clear Files", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Initialize
    rag = RAGSystem()
    
    # Sidebar
    show_sidebar()
    
    # Main header
    st.title("🤖 OmniBot AI")
    st.markdown("---")
    
    # Tabs
    tab_chat, tab_docs, tab_code = st.tabs(["Chat", "Documents", "Code"])
    
    # TAB 1: CHAT
    with tab_chat:
        st.subheader("Chat with AI")
        
        # Display chat
        for msg in st.session_state.conversation[-20:]:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="ai-message">
                    <strong>AI:</strong> {msg['content']}
                </div>
                """, unsafe_allow_html=True)
        
        # Input
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input("Your message", key="chat_input", label_visibility="collapsed")
        with col2:
            send = st.button("Send", use_container_width=True)
        
        if send and user_input:
            llm = ModelManager.get_model()
            if not llm:
                st.error("Please add API key in sidebar")
                return
            
            # Add user message
            st.session_state.conversation.append({
                'role': 'user',
                'content': user_input,
                'time': datetime.now().isoformat()
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Simple conversation context
                    recent = st.session_state.conversation[-4:]
                    messages = [SystemMessage(content="You are a helpful AI assistant.")]
                    
                    for msg in recent:
                        if msg['role'] == 'user':
                            messages.append(HumanMessage(content=msg['content']))
                        else:
                            messages.append(AIMessage(content=msg['content']))
                    
                    prompt = ChatPromptTemplate.from_messages(messages)
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({})
                    
                    # Add AI response
                    st.session_state.conversation.append({
                        'role': 'assistant',
                        'content': response,
                        'time': datetime.now().isoformat()
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # TAB 2: DOCUMENTS
    with tab_docs:
        st.subheader("Document Intelligence")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Upload
            uploaded = st.file_uploader(
                "Upload PDF, TXT, or CSV files",
                type=["pdf", "txt", "csv"],
                accept_multiple_files=True
            )
            
            if uploaded and st.button("Process"):
                with st.spinner("Processing..."):
                    if rag.process_files(uploaded):
                        st.success(f"Processed {len(uploaded)} files")
                    else:
                        st.error("Failed to process files")
            
            # Document Q&A
            if st.session_state.vectorstore:
                st.markdown("### Ask about your documents")
                question = st.text_input("Question about documents")
                
                if question and st.button("Search"):
                    with st.spinner("Searching..."):
                        context = rag.search(question)
                        answer = rag.answer_with_context(question, context)
                        
                        st.markdown("**Answer:**")
                        st.write(answer)
                        
                        if context:
                            with st.expander("Sources"):
                                for i, doc in enumerate(context):
                                    st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                                    st.caption(doc.page_content[:200] + "...")
        
        with col2:
            # File list
            if st.session_state.processed_files:
                st.markdown("### Processed Files")
                for f in st.session_state.processed_files:
                    st.write(f"• {f}")
            
            # Quick stats
            if st.session_state.vectorstore:
                st.markdown("### Document Stats")
                st.metric("Total Files", len(st.session_state.processed_files))
                st.metric("Search Ready", "✅" if st.session_state.vectorstore else "❌")
    
    # TAB 3: CODE
    with tab_code:
        st.subheader("Code Assistant")
        
        # Task input
        task = st.text_area(
            "Describe what you want to code:",
            height=100,
            placeholder="e.g., Create a Python function that validates email addresses..."
        )
        
        # Options
        col1, col2, col3 = st.columns(3)
        with col1:
            lang = st.selectbox("Language", ["Python", "JavaScript", "SQL", "Java", "HTML/CSS"])
        with col2:
            level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        with col3:
            include_explanation = st.checkbox("Include explanation", value=True)
        
        if task and st.button("Generate Code"):
            llm = ModelManager.get_model()
            if not llm:
                st.error("Please add API key")
                return
            
            with st.spinner("Coding..."):
                prompt = f"""Create {lang} code for this task: {task}
                
                Level: {level}
                Include explanation: {include_explanation}
                
                Provide clean, working code with comments."""
                
                try:
                    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
                    code = chain.invoke({})
                    
                    st.markdown("### Generated Code")
                    st.code(code, language=lang.lower())
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================================================
# RUN
# ============================================================================
if __name__ == "__main__":
    main()
