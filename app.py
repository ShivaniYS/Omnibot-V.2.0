import streamlit as st
import os
import tempfile
import json
from datetime import datetime

# Core AI Libraries
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

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
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1 {
        color: #1a73e8;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    h2 {
        color: #333;
        margin-top: 1.5rem;
    }
    
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
    
    .user-message {
        background-color: #e8f0fe;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #1a73e8;
    }
    
    .ai-message {
        background-color: #f8f9fa;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 10px 0;
        border-left: 4px solid #34a853;
    }
    
    .file-upload-area {
        border: 2px dashed #1a73e8;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9fa;
        margin: 20px 0;
    }
    
    .stButton > button {
        background-color: #1a73e8;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: 500;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    defaults = {
        'conversation': [],
        'processed_files': [],
        'api_keys': {
            'GROQ_API_KEY': "",
            'OPENAI_API_KEY': ""
        },
        'model': 'groq-llama-70b',
        'temperature': 0.7,
        'max_tokens': 1024,
        'code_history': [],
        'document_content': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# DOCUMENT PROCESSING (SIMPLIFIED - NO PyPDF DEPENDENCY)
# ============================================================================
def extract_text_from_file(file_path, file_type):
    """Extract text from files without PyPDF dependency"""
    try:
        if file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type == 'csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string()
        elif file_type == 'pdf':
            # Try different PDF extraction methods
            try:
                # Method 1: Try pypdf (most common)
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                try:
                    # Method 2: Try pdfplumber (alternative)
                    import pdfplumber
                    text = ""
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            text += page.extract_text() + "\n"
                    return text
                except ImportError:
                    # Method 3: Try PyPDF2 (another alternative)
                    try:
                        import PyPDF2
                        text = ""
                        with open(file_path, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            for page in pdf_reader.pages:
                                text += page.extract_text() + "\n"
                        return text
                    except ImportError:
                        return None
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

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
# SIDEBAR
# ============================================================================
def show_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # API Keys
        st.subheader("API Keys")
        groq_key = st.text_input(
            "Groq API Key",
            value=st.session_state.api_keys['GROQ_API_KEY'],
            type="password",
            placeholder="sk-..."
        )
        
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_keys['OPENAI_API_KEY'],
            type="password",
            placeholder="sk-..."
        )
        
        if groq_key != st.session_state.api_keys['GROQ_API_KEY']:
            st.session_state.api_keys['GROQ_API_KEY'] = groq_key
        
        if openai_key != st.session_state.api_keys['OPENAI_API_KEY']:
            st.session_state.api_keys['OPENAI_API_KEY'] = openai_key
        
        # Model Selection
        st.subheader("Model")
        available = ModelManager.get_available()
        if available:
            model_names = {
                'groq-llama-70b': 'Llama 70B (Recommended)',
                'groq-llama-8b': 'Llama 8B (Fast)',
                'openai-gpt4': 'GPT-4 (Most Capable)',
                'openai-gpt35': 'GPT-3.5 (Economical)'
            }
            selected = st.selectbox(
                "Choose model",
                available,
                format_func=lambda x: model_names.get(x, x)
            )
            st.session_state.model = selected
            
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
            with col2:
                st.session_state.max_tokens = st.slider("Max Tokens", 256, 4096, 1024, 256)
        else:
            st.info("Add API keys to enable models")
        
        # Stats
        st.subheader("Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chat Messages", len(st.session_state.conversation))
        with col2:
            st.metric("Files", len(st.session_state.processed_files))
        
        # Clear buttons
        st.subheader("Management")
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()
        
        if st.button("Clear Files", use_container_width=True):
            st.session_state.processed_files = []
            st.session_state.document_content = {}
            st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Show sidebar
    show_sidebar()
    
    # Main header
    st.title("🤖 OmniBot AI")
    st.markdown("An intelligent assistant for chat, documents, and code generation")
    
    # Check if dependencies are installed
    try:
        import pypdf
        pdf_support = True
    except ImportError:
        try:
            import pdfplumber
            pdf_support = True
        except ImportError:
            try:
                import PyPDF2
                pdf_support = True
            except ImportError:
                pdf_support = False
                st.warning("""
                ⚠️ **PDF support is limited** - Install one of these packages:
                ```
                pip install pypdf
                # or
                pip install pdfplumber
                # or
                pip install PyPDF2
                ```
                """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "💻 Code"])
    
    # TAB 1: CHAT
    with tab1:
        st.header("Chat with AI")
        
        # Check API
        if not ModelManager.get_available():
            st.info("👋 Welcome! Please add an API key in the sidebar to start chatting.")
        
        # Chat display
        for msg in st.session_state.conversation[-10:]:
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
        
        # Chat input
        user_input = st.text_input(
            "Type your message...",
            key="chat_input",
            label_visibility="collapsed",
            placeholder="Ask me anything..."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send = st.button("Send", use_container_width=True)
        
        if send and user_input:
            llm = ModelManager.get_model()
            if not llm:
                st.error("Please add an API key in the sidebar")
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
                    # Build messages
                    messages = [SystemMessage(content="You are a helpful AI assistant.")]
                    
                    # Add recent conversation
                    for msg in st.session_state.conversation[-4:]:
                        if msg['role'] == 'user':
                            messages.append(HumanMessage(content=msg['content']))
                        else:
                            messages.append(AIMessage(content=msg['content']))
                    
                    # Create chain
                    prompt = ChatPromptTemplate.from_messages(messages)
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({})
                    
                    # Add response
                    st.session_state.conversation.append({
                        'role': 'assistant',
                        'content': response,
                        'time': datetime.now().isoformat()
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # TAB 2: DOCUMENTS
    with tab2:
        st.header("Document Intelligence")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # File upload
            st.markdown("""
            <div class="file-upload-area">
                <h3>📁 Drag and drop files here</h3>
                <p>Limit 200MB per file • PDF, TXT, CSV</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_files = st.file_uploader(
                "",
                type=["pdf", "txt", "csv"],
                accept_multiple_files=True,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                st.success(f"Selected {len(uploaded_files)} file(s)")
                
                # Show file list
                for file in uploaded_files:
                    size_kb = file.size / 1024
                    st.write(f"**{file.name}** ({size_kb:.1f} KB)")
                
                if st.button("📄 Process Files", type="primary"):
                    processed_count = 0
                    
                    for uploaded_file in uploaded_files:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        
                        # Create temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as f:
                            f.write(uploaded_file.getbuffer())
                            temp_path = f.name
                        
                        try:
                            # Extract text
                            text = extract_text_from_file(temp_path, file_ext)
                            
                            if text:
                                # Store content
                                file_id = str(uuid.uuid4())[:8]
                                st.session_state.document_content[file_id] = {
                                    'name': uploaded_file.name,
                                    'text': text[:5000],  # Store first 5000 chars
                                    'type': file_ext,
                                    'size': uploaded_file.size,
                                    'processed': datetime.now().isoformat()
                                }
                                
                                if uploaded_file.name not in st.session_state.processed_files:
                                    st.session_state.processed_files.append(uploaded_file.name)
                                
                                st.success(f"✅ Processed: {uploaded_file.name}")
                                processed_count += 1
                            else:
                                if file_ext == 'pdf' and not pdf_support:
                                    st.error(f"❌ {uploaded_file.name}: Install PDF package (see warning above)")
                                else:
                                    st.error(f"❌ Failed to process: {uploaded_file.name}")
                        
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        
                        finally:
                            # Cleanup
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                    
                    if processed_count > 0:
                        st.balloons()
                        st.success(f"Successfully processed {processed_count} file(s)")
            
            # Document Q&A
            if st.session_state.document_content:
                st.markdown("---")
                st.subheader("Ask Questions")
                
                question = st.text_input("Ask about your documents:", key="doc_question")
                
                if question and st.button("🔍 Search"):
                    # Collect all document content
                    all_text = ""
                    for file_id, doc in st.session_state.document_content.items():
                        all_text += f"\n\n--- {doc['name']} ---\n{doc['text']}"
                    
                    if all_text:
                        llm = ModelManager.get_model()
                        if llm:
                            with st.spinner("Searching..."):
                                prompt = f"""Based on these documents, answer the question:

Documents:
{all_text[:4000]}...

Question: {question}

If the answer isn't in the documents, say "This information is not in the provided documents"."""
                                
                                try:
                                    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
                                    answer = chain.invoke({})
                                    
                                    st.markdown("**Answer:**")
                                    st.write(answer)
                                    
                                    # Show which documents were referenced
                                    with st.expander("📚 Document References"):
                                        for file_id, doc in st.session_state.document_content.items():
                                            if doc['name'].lower() in answer.lower() or doc['text'][:100].lower() in answer.lower():
                                                st.write(f"• **{doc['name']}** ({doc['type'].upper()})")
                                
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                        else:
                            st.error("Please add an API key")
        
        with col2:
            # Processed files list
            if st.session_state.processed_files:
                st.subheader("📂 Your Files")
                
                for file_name in st.session_state.processed_files:
                    # Find file info
                    file_info = None
                    for doc in st.session_state.document_content.values():
                        if doc['name'] == file_name:
                            file_info = doc
                            break
                    
                    if file_info:
                        st.write(f"**{file_name}**")
                        st.caption(f"{file_info['type'].upper()} • {file_info['size']:,} bytes")
                        
                        # Preview
                        with st.expander("Preview"):
                            preview = file_info['text'][:300] + "..." if len(file_info['text']) > 300 else file_info['text']
                            st.text(preview)
                    else:
                        st.write(f"• {file_name}")
                    
                    st.divider()
                
                # Quick stats
                st.subheader("📊 Stats")
                total_size = sum(doc['size'] for doc in st.session_state.document_content.values())
                st.metric("Total Files", len(st.session_state.processed_files))
                st.metric("Total Size", f"{total_size/1024:.1f} KB")
            else:
                st.info("No files processed yet")
                st.markdown("""
                **How to use:**
                1. Upload PDF, TXT, or CSV files
                2. Click "Process Files"
                3. Ask questions about your documents
                
                **Note:** For PDF support, install:
                ```bash
                pip install pypdf
                ```
                """)
    
    # TAB 3: CODE
    with tab3:
        st.header("Code Assistant")
        
        # Task input
        task = st.text_area(
            "Describe what you want to code:",
            height=100,
            placeholder="Example: 'Create a Python function to validate email addresses'"
        )
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Language", ["Python", "JavaScript", "HTML/CSS", "SQL", "Java", "C++"])
        with col2:
            level = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        
        if task and st.button("Generate Code", type="primary"):
            llm = ModelManager.get_model()
            if not llm:
                st.error("Please add an API key")
                return
            
            with st.spinner("Coding..."):
                prompt = f"""Create {language} code for this task:

{task}

Level: {level}
Include comments and explanation.

Provide clean, working code."""
                
                try:
                    chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
                    code = chain.invoke({})
                    
                    st.markdown("### Generated Code")
                    st.code(code, language=language.lower())
                    
                    # Store in history
                    st.session_state.code_history.append({
                        'task': task,
                        'language': language,
                        'code': code,
                        'time': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Code history
        if st.session_state.code_history:
            st.markdown("---")
            st.subheader("Recent Code")
            
            for item in reversed(st.session_state.code_history[-3:]):
                with st.expander(f"{item['language']}: {item['task'][:50]}..."):
                    st.code(item['code'], language=item['language'].lower())

# ============================================================================
# REQUIREMENTS FOR DEPLOYMENT
# ============================================================================
def show_requirements():
    st.sidebar.markdown("---")
    with st.sidebar.expander("📦 Deployment Requirements"):
        st.markdown("""
        **For Cloud Deployment:**
        
        Add to `requirements.txt`:
        ```
        streamlit>=1.28.0
        langchain-groq>=0.1.0
        langchain-openai>=0.0.5
        pypdf>=3.0.0  # For PDF support
        pandas>=2.0.0  # For CSV support
        python-dotenv>=1.0.0
        ```
        
        **Or install all at once:**
        ```bash
        pip install streamlit langchain-groq langchain-openai pypdf pandas python-dotenv
        ```
        
        **For PDF support (choose one):**
        - `pypdf` (recommended)
        - `pdfplumber` (more features)
        - `PyPDF2` (legacy)
        """)

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    show_requirements()
    main()
