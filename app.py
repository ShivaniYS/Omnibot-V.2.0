import streamlit as st
import os
import tempfile
import json
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# RAG Components - simplified imports
try:
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
    RAG_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some RAG components not available: {e}")
    RAG_AVAILABLE = False

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="OmniBot AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean CSS
st.markdown("""
<style>
    /* Clean up Streamlit defaults */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1a73e8;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    h2 {
        color: #333;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #555;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #f8f9fa;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
        border-radius: 6px;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1a73e8 !important;
        color: white !important;
    }
    
    /* Message styles */
    .user-message {
        background-color: #e8f0fe;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #1a73e8;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .ai-message {
        background-color: #f8f9fa;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #34a853;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* File uploader styling */
    .st-emotion-cache-1r4v9p6 {
        border: 2px dashed #1a73e8;
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0d62d9;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    defaults = {
        'conversation': [],
        'vectorstore': None,
        'processed_files': [],
        'api_keys': {
            'GROQ_API_KEY': "",
            'OPENAI_API_KEY': ""
        },
        'model': 'groq-llama-70b',
        'temperature': 0.7,
        'max_tokens': 1024,
        'code_history': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# MODEL MANAGER
# ============================================================================
class ModelManager:
    @staticmethod
    def get_model():
        model_id = st.session_state.model
        api_keys = st.session_state.api_keys
        
        try:
            if model_id.startswith('groq'):
                api_key = api_keys.get('GROQ_API_KEY')
                if not api_key:
                    st.warning("Please enter your Groq API key in the sidebar")
                    return None
                
                model_name = 'llama-3.3-70b-versatile' if '70b' in model_id else 'llama-3.1-8b-instant'
                return ChatGroq(
                    groq_api_key=api_key,
                    model_name=model_name,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
            else:
                api_key = api_keys.get('OPENAI_API_KEY')
                if not api_key:
                    st.warning("Please enter your OpenAI API key in the sidebar")
                    return None
                
                model_name = 'gpt-4-turbo-preview' if 'gpt4' in model_id else 'gpt-3.5-turbo'
                return ChatOpenAI(
                    api_key=api_key,
                    model_name=model_name,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    @staticmethod
    def get_available():
        available = []
        if st.session_state.api_keys.get('GROQ_API_KEY'):
            available.extend(['groq-llama-70b', 'groq-llama-8b'])
        if st.session_state.api_keys.get('OPENAI_API_KEY'):
            available.extend(['openai-gpt4', 'openai-gpt35'])
        return available

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
class DocumentProcessor:
    def __init__(self):
        if RAG_AVAILABLE:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
        else:
            st.warning("RAG features are disabled due to missing dependencies")
    
    def process_files(self, uploaded_files):
        if not RAG_AVAILABLE:
            st.error("RAG features require additional packages. Please install: pip install faiss-cpu sentence-transformers pypdf")
            return False
        
        documents = []
        temp_files = []
        
        try:
            for uploaded_file in uploaded_files:
                file_name = uploaded_file.name
                file_ext = file_name.split('.')[-1].lower()
                
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as f:
                    f.write(uploaded_file.getbuffer())
                    temp_path = f.name
                    temp_files.append(temp_path)
                
                # Load based on file type
                try:
                    if file_ext == 'pdf':
                        loader = PyPDFLoader(temp_path)
                    elif file_ext == 'txt':
                        loader = TextLoader(temp_path)
                    elif file_ext == 'csv':
                        loader = CSVLoader(temp_path)
                    else:
                        st.warning(f"Unsupported file type: {file_ext}")
                        continue
                    
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            'source': file_name,
                            'file_type': file_ext,
                            'processed_at': datetime.now().isoformat()
                        })
                    documents.extend(docs)
                    st.success(f"✓ Loaded {file_name}")
                    
                except Exception as e:
                    st.error(f"Failed to load {file_name}: {str(e)}")
            
            if documents:
                # Split documents
                splits = self.splitter.split_documents(documents)
                
                # Create vector store
                vectorstore = FAISS.from_documents(splits, self.embeddings)
                
                # Store in session
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = [f.name for f in uploaded_files]
                
                # Show stats
                st.success(f"✅ Successfully processed {len(uploaded_files)} file(s)")
                st.info(f"• Created {len(splits)} text chunks\n• Ready for Q&A")
                return True
            else:
                st.warning("No documents were successfully loaded")
                return False
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            return False
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def search_documents(self, query, k=3):
        if not st.session_state.vectorstore:
            return []
        
        try:
            return st.session_state.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

# ============================================================================
# SIDEBAR
# ============================================================================
def show_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Keys Section
        st.subheader("API Configuration")
        
        with st.expander("API Keys", expanded=True):
            groq_key = st.text_input(
                "Groq API Key",
                value=st.session_state.api_keys['GROQ_API_KEY'],
                type="password",
                placeholder="Enter your Groq API key"
            )
            
            openai_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.api_keys['OPENAI_API_KEY'],
                type="password",
                placeholder="Enter your OpenAI API key"
            )
            
            if groq_key != st.session_state.api_keys['GROQ_API_KEY']:
                st.session_state.api_keys['GROQ_API_KEY'] = groq_key
            
            if openai_key != st.session_state.api_keys['OPENAI_API_KEY']:
                st.session_state.api_keys['OPENAI_API_KEY'] = openai_key
        
        # Model Selection
        st.subheader("Model Settings")
        
        available_models = ModelManager.get_available()
        if available_models:
            model_display = {
                'groq-llama-70b': 'Groq Llama 3 70B',
                'groq-llama-8b': 'Groq Llama 3 8B',
                'openai-gpt4': 'OpenAI GPT-4',
                'openai-gpt35': 'OpenAI GPT-3.5'
            }
            
            selected = st.selectbox(
                "Select Model",
                options=available_models,
                format_func=lambda x: model_display.get(x, x),
                index=0
            )
            
            if selected != st.session_state.model:
                st.session_state.model = selected
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.temperature = st.slider(
                    "Temperature",
                    0.0, 1.0, 0.7, 0.1,
                    help="Higher = more creative, Lower = more focused"
                )
            
            with col2:
                st.session_state.max_tokens = st.slider(
                    "Max Tokens",
                    256, 4096, 1024, 256,
                    help="Maximum length of responses"
                )
        else:
            st.info("👆 Add API keys to enable models")
        
        # Stats Section
        st.subheader("📊 Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chat Messages", len(st.session_state.conversation))
        with col2:
            st.metric("Processed Files", len(st.session_state.processed_files))
        
        # Management Section
        st.subheader("🔄 Management")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()
        
        if st.button("📂 Clear Documents", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.processed_files = []
            st.rerun()
        
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['api_keys']:
                    del st.session_state[key]
            init_session_state()
            st.rerun()
        
        st.markdown("---")
        st.caption("OmniBot AI v1.0")

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Initialize processor
    doc_processor = DocumentProcessor()
    
    # Show sidebar
    show_sidebar()
    
    # Main header
    st.title("🤖 OmniBot AI")
    st.markdown("An intelligent assistant for chat, documents, and code generation")
    st.markdown("---")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "💻 Code"])
    
    # TAB 1: CHAT
    with tab1:
        st.header("Chat with AI")
        
        # Check API keys
        if not ModelManager.get_available():
            st.warning("⚠️ Please add an API key in the sidebar to start chatting")
        
        # Display chat history
        chat_container = st.container(height=500, border=True)
        
        with chat_container:
            if not st.session_state.conversation:
                st.info("Start a conversation by typing a message below!")
            
            for msg in st.session_state.conversation[-20:]:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>You:</strong><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ai-message">
                        <strong>Assistant:</strong><br>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([6, 1])
        with col1:
            user_input = st.text_input(
                "Type your message here...",
                key="chat_input",
                label_visibility="collapsed",
                placeholder="Ask me anything..."
            )
        
        with col2:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        if send_button and user_input:
            llm = ModelManager.get_model()
            
            if not llm:
                st.error("Please configure an API key in the sidebar")
                st.stop()
            
            # Add user message
            st.session_state.conversation.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Prepare conversation context
                    recent_msgs = st.session_state.conversation[-6:]  # Last 6 messages
                    
                    # Build messages list
                    messages = [SystemMessage(content="You are OmniBot, a helpful AI assistant. Be concise, accurate, and friendly.")]
                    
                    for msg in recent_msgs:
                        if msg['role'] == 'user':
                            messages.append(HumanMessage(content=msg['content']))
                        else:
                            messages.append(AIMessage(content=msg['content']))
                    
                    # Create and run chain
                    prompt = ChatPromptTemplate.from_messages(messages)
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({})
                    
                    # Add assistant response
                    st.session_state.conversation.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.session_state.conversation.append({
                        'role': 'assistant',
                        'content': f"I encountered an error: {str(e)}",
                        'timestamp': datetime.now().isoformat()
                    })
                    st.rerun()
    
    # TAB 2: DOCUMENTS
    with tab2:
        st.header("Document Intelligence")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # File upload section
            st.subheader("Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Choose files to upload",
                type=["pdf", "txt", "csv"],
                accept_multiple_files=True,
                help="Upload PDF, TXT, or CSV files for analysis"
            )
            
            if uploaded_files:
                st.info(f"Selected {len(uploaded_files)} file(s)")
                
                # Show file list
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size:,} bytes)")
                
                if st.button("🚀 Process Files", type="primary"):
                    with st.spinner("Processing documents..."):
                        success = doc_processor.process_files(uploaded_files)
                        if success:
                            st.balloons()
            
            # Document Q&A Section
            if st.session_state.vectorstore:
                st.subheader("Ask Questions")
                
                question = st.text_input(
                    "Ask about your documents",
                    placeholder="What information are you looking for?"
                )
                
                if question and st.button("🔍 Search & Answer"):
                    with st.spinner("Searching documents..."):
                        # Search for relevant documents
                        context_chunks = doc_processor.search_documents(question, k=3)
                        
                        if context_chunks:
                            # Prepare context for LLM
                            context_text = "\n\n---\n\n".join([
                                f"**From {chunk.metadata.get('source', 'Document')}:**\n{chunk.page_content[:500]}..."
                                for chunk in context_chunks
                            ])
                            
                            # Get LLM response
                            llm = ModelManager.get_model()
                            if llm:
                                prompt = f"""Based on the following document excerpts, answer the question.

Document Content:
{context_text}

Question: {question}

Answer based on the documents. If the information isn't available, say so clearly."""
                                
                                try:
                                    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
                                    answer = chain.invoke({})
                                    
                                    st.markdown("### 📝 Answer")
                                    st.write(answer)
                                    
                                    # Show sources
                                    with st.expander("📚 Source Documents"):
                                        for i, chunk in enumerate(context_chunks):
                                            source = chunk.metadata.get('source', 'Unknown')
                                            st.write(f"**Source {i+1}:** {source}")
                                            st.caption(chunk.page_content[:300] + "...")
                                            st.divider()
                                except Exception as e:
                                    st.error(f"Error generating answer: {str(e)}")
                            else:
                                st.error("Please configure an API key")
                        else:
                            st.warning("No relevant documents found")
        
        with col2:
            # File list and stats
            st.subheader("📂 Processed Files")
            
            if st.session_state.processed_files:
                st.success(f"You have {len(st.session_state.processed_files)} processed file(s)")
                
                for file in st.session_state.processed_files:
                    st.markdown(f"""
                    <div class="metric-card">
                        📄 {file}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Quick stats
                st.subheader("📊 Document Stats")
                
                if st.session_state.vectorstore:
                    try:
                        # Get some stats
                        index = st.session_state.vectorstore.index
                        if hasattr(index, 'ntotal'):
                            st.metric("Vectors in Index", index.ntotal)
                    except:
                        pass
                    
                    st.metric("Search Ready", "✅")
                else:
                    st.metric("Search Ready", "❌")
            else:
                st.info("No documents processed yet")
                st.markdown("""
                **Supported formats:**
                - PDF documents
                - Text files (.txt)
                - CSV files
                
                **After uploading:**
                1. Click "Process Files"
                2. Ask questions about your documents
                3. Get answers with citations
                """)
    
    # TAB 3: CODE
    with tab3:
        st.header("Code Assistant")
        
        # Task description
        task = st.text_area(
            "Describe what you want to code:",
            height=150,
            placeholder="Example: 'Create a Python function to validate email addresses'\nOr: 'Write a React component for a login form'",
            help="Describe your coding task in detail"
        )
        
        # Options
        col1, col2, col3 = st.columns(3)
        with col1:
            language = st.selectbox(
                "Programming Language",
                ["Python", "JavaScript", "Java", "HTML/CSS", "SQL", "TypeScript", "C++", "Go"],
                index=0
            )
        
        with col2:
            include_tests = st.checkbox("Include tests", value=True)
            include_comments = st.checkbox("Include comments", value=True)
        
        with col3:
            complexity = st.select_slider(
                "Complexity",
                options=["Simple", "Intermediate", "Advanced"],
                value="Intermediate"
            )
        
        if task and st.button("✨ Generate Code", type="primary"):
            llm = ModelManager.get_model()
            
            if not llm:
                st.error("Please configure an API key in the sidebar")
                st.stop()
            
            with st.spinner("Writing code..."):
                try:
                    # Build prompt
                    prompt = f"""You are an expert {language} developer.

Task: {task}

Requirements:
- Language: {language}
- Complexity level: {complexity}
- Include tests: {include_tests}
- Include comments: {include_comments}

Provide:
1. Complete, runnable code
2. Brief explanation of the solution
3. {f"Test cases/examples" if include_tests else "Usage examples"}
4. Any important notes

Format with proper code blocks and clear structure."""

                    # Generate code
                    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
                    code_response = chain.invoke({})
                    
                    # Display results
                    st.markdown("### Generated Code")
                    
                    # Use code block with language
                    language_lower = language.lower()
                    if language_lower == "html/css":
                        language_lower = "html"
                    
                    st.code(code_response, language=language_lower)
                    
                    # Store in history
                    st.session_state.code_history.append({
                        'task': task,
                        'language': language,
                        'response': code_response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Show success
                    st.success("Code generated successfully!")
                    
                    # Copy button
                    st.download_button(
                        label="📥 Download Code",
                        data=code_response,
                        file_name=f"code_{language.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"Error generating code: {str(e)}")
        
        # Code history
        if st.session_state.code_history:
            st.subheader("📚 Recent Code Generations")
            
            for i, item in enumerate(reversed(st.session_state.code_history[-3:])):
                with st.expander(f"{item['language']}: {item['task'][:50]}..."):
                    st.write(f"**Task:** {item['task']}")
                    st.write(f"**Generated at:** {item['timestamp'][:19]}")
                    st.code(item['response'][:500] + "..." if len(item['response']) > 500 else item['response'],
                           language=item['language'].lower())

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
