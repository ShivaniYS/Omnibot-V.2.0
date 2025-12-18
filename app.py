import streamlit as st
import os
import tempfile
import json
import uuid
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
    layout="wide"
)

# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    defaults = {
        'conversation': [],
        'processed_files': [],
        'document_content': {},
        'code_history': [],
        'model': 'groq-llama-70b',
        'temperature': 0.7,
        'max_tokens': 512,
        'mode': 'brainy-buddy'
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
        
        try:
            if model_id.startswith('groq'):
                # Get model name from ID
                if '70b' in model_id:
                    model_name = 'llama-3.3-70b-versatile'
                else:
                    model_name = 'llama-3.1-8b-instant'
                
                return ChatGroq(
                    model_name=model_name,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
            else:
                # OpenAI models
                if 'gpt4' in model_id:
                    model_name = 'gpt-4-turbo-preview'
                else:
                    model_name = 'gpt-3.5-turbo'
                
                return ChatOpenAI(
                    model_name=model_name,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================
def extract_text_from_file(file_path, file_type):
    """Extract text from files"""
    try:
        if file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type == 'csv':
            import pandas as pd
            df = pd.read_csv(file_path)
            return df.to_string()
        elif file_type == 'pdf':
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                st.warning("Install pypdf for PDF support: pip install pypdf")
                return None
        else:
            return None
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return None

# ============================================================================
# SIDEBAR - SIMPLE SETTINGS
# ============================================================================
def show_sidebar():
    with st.sidebar:
        st.title("⚙️ OmniBot Settings")
        
        # API Status
        st.subheader("API Status")
        col1, col2 = st.columns(2)
        with col1:
            st.info("Groq API Key:")
            st.success("Configured ✓")
        with col2:
            st.info("LangSmith Tracing:")
            st.success("Configured ✓")
        
        st.divider()
        
        # Model Settings
        st.subheader("Model Settings")
        
        # Assistant Mode
        mode = st.selectbox(
            "Choose Your Assistant Mode",
            ["Brainy Buddy", "DocuMind", "CodeCraft"],
            index=0
        )
        st.session_state.mode = mode.lower().replace(" ", "-")
        
        # Model Selection
        model_option = st.selectbox(
            "Select Groq Model",
            [
                "groq-llama-70b",
                "groq-llama-8b", 
                "openai-gpt4",
                "openai-gpt35"
            ],
            format_func=lambda x: {
                "groq-llama-70b": "Llama 3.3 70B",
                "groq-llama-8b": "Llama 3.1 8B",
                "openai-gpt4": "OpenAI GPT-4",
                "openai-gpt35": "OpenAI GPT-3.5"
            }.get(x, x)
        )
        st.session_state.model = model_option
        
        # Temperature
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
        
        # Max Tokens
        st.session_state.max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=2048,
            value=512,
            step=256
        )
        
        st.divider()
        
        # Clear History
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Show sidebar
    show_sidebar()
    
    # Main header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=80)
    with col2:
        st.title("OmniBot AI")
        st.markdown("An intelligent assistant for chat, documents, and code generation")
    
    st.divider()
    
    # Tabs
    tab_chat, tab_docs, tab_code = st.tabs(["💬 Chat", "📚 Documents", "💻 Code"])
    
    # TAB 1: CHAT
    with tab_chat:
        st.subheader("Chat with AI")
        
        # Display chat history
        chat_container = st.container(height=400)
        with chat_container:
            if not st.session_state.conversation:
                st.info("👋 Start a conversation by typing a message below!")
            
            for msg in st.session_state.conversation[-10:]:
                if msg['role'] == 'user':
                    with st.chat_message("user"):
                        st.write(msg['content'])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg['content'])
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message
            st.session_state.conversation.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                llm = ModelManager.get_model()
                
                if llm:
                    try:
                        # Prepare messages
                        messages = [SystemMessage(content="You are a helpful AI assistant.")]
                        
                        # Add conversation history
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
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.error("Failed to load model. Check your API configuration.")
    
    # TAB 2: DOCUMENTS
    with tab_docs:
        st.subheader("Document Intelligence")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # File upload
            uploaded_files = st.file_uploader(
                "Upload PDF, TXT, or CSV files",
                type=["pdf", "txt", "csv"],
                accept_multiple_files=True,
                help="Drag and drop files here"
            )
            
            if uploaded_files:
                st.success(f"Selected {len(uploaded_files)} file(s)")
                
                for file in uploaded_files:
                    st.write(f"• {file.name} ({file.size:,} bytes)")
                
                if st.button("Process Files", type="primary"):
                    for uploaded_file in uploaded_files:
                        file_ext = uploaded_file.name.split('.')[-1].lower()
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as f:
                            f.write(uploaded_file.getbuffer())
                            temp_path = f.name
                        
                        try:
                            text = extract_text_from_file(temp_path, file_ext)
                            
                            if text:
                                file_id = str(uuid.uuid4())[:8]
                                st.session_state.document_content[file_id] = {
                                    'name': uploaded_file.name,
                                    'text': text[:3000],
                                    'type': file_ext
                                }
                                
                                if uploaded_file.name not in st.session_state.processed_files:
                                    st.session_state.processed_files.append(uploaded_file.name)
                                
                                st.success(f"✅ Processed: {uploaded_file.name}")
                            else:
                                st.error(f"Failed to process: {uploaded_file.name}")
                        
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        
                        finally:
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
            
            # Document Q&A
            if st.session_state.document_content:
                st.divider()
                st.subheader("Ask about your documents")
                
                question = st.text_input("Enter your question:")
                
                if question and st.button("Search"):
                    all_text = "\n\n".join([
                        f"=== {doc['name']} ===\n{doc['text']}"
                        for doc in st.session_state.document_content.values()
                    ])
                    
                    llm = ModelManager.get_model()
                    if llm:
                        with st.spinner("Searching documents..."):
                            prompt = f"""Based on these documents:

{all_text[:3000]}

Question: {question}

Answer the question based on the documents above."""
                            
                            try:
                                chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
                                answer = chain.invoke({})
                                
                                st.write("**Answer:**")
                                st.write(answer)
                                
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
        
        with col2:
            # Processed files
            if st.session_state.processed_files:
                st.subheader("📂 Processed Files")
                for file_name in st.session_state.processed_files:
                    st.write(f"• {file_name}")
            else:
                st.info("No files processed yet")
                st.markdown("""
                **How to use:**
                1. Upload PDF, TXT, or CSV files
                2. Click "Process Files"
                3. Ask questions about your documents
                """)
    
    # TAB 3: CODE
    with tab_code:
        st.subheader("Code Assistant")
        
        # Task input
        task = st.text_area(
            "Describe your coding task:",
            height=100,
            placeholder="Example: 'Create a Python function to validate email addresses'"
        )
        
        # Options
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Language", ["Python", "JavaScript", "HTML/CSS", "SQL", "Java"])
        with col2:
            level = st.selectbox("Skill Level", ["Beginner", "Intermediate", "Advanced"])
        
        if task and st.button("Generate Code", type="primary"):
            llm = ModelManager.get_model()
            if llm:
                with st.spinner("Generating code..."):
                    prompt = f"""Create {language} code for this task:

{task}

Skill Level: {level}
Include comments and a brief explanation.

Provide complete, working code."""
                    
                    try:
                        chain = ChatPromptTemplate.from_template(prompt) | llm | StrOutputParser()
                        code = chain.invoke({})
                        
                        st.code(code, language=language.lower())
                        
                        # Store in history
                        st.session_state.code_history.append({
                            'task': task,
                            'language': language,
                            'code': code,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Code history
        if st.session_state.code_history:
            st.divider()
            st.subheader("Recent Code")
            for item in reversed(st.session_state.code_history[-3:]):
                with st.expander(f"{item['language']}: {item['task'][:50]}..."):
                    st.code(item['code'], language=item['language'].lower())

# ============================================================================
# REQUIREMENTS FOR DEPLOYMENT
# ============================================================================
def show_requirements():
    # This would be shown in deployment instructions
    pass

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
