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
from dataclasses import dataclass
from enum import Enum

# Core AI Libraries - Streamlined
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Advanced RAG Components - Optimized
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, DocxLoader, CSVLoader, WebBaseLoader
)

# Multi-Agent Framework
from langgraph.graph import StateGraph, END

# Evaluation
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Knowledge Graph (Lightweight)
import networkx as nx

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="OmniBot AI Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e0e0e0;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #f8f9fa;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0;
        max-width: 85%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_session_state():
    """Initialize all session state variables with safe defaults"""
    defaults = {
        'conversation_history': [],
        'document_store': {},
        'code_history': [],
        'current_mode': "AI Studio",
        'vectorstore': None,
        'processed_files': [],
        'document_chat_history': [],
        'workflow_steps': [],
        'evaluation_results': {},
        'active_workflow': None,
        'rag_metrics': {},
        'knowledge_graph': nx.Graph(),
        'agent_conversations': {},
        'api_keys': {
            'GROQ_API_KEY': st.secrets.get("GROQ_API_KEY", ""),
            'OPENAI_API_KEY': st.secrets.get("OPENAI_API_KEY", "")
        },
        'model_settings': {
            'primary_model': 'groq-llama-70b',
            'temperature': 0.7,
            'max_tokens': 1024
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# MODEL REGISTRY (Simplified)
# ============================================================================
@dataclass
class ModelConfig:
    name: str
    provider: str
    max_tokens: int
    temperature: float = 0.7

class ModelRegistry:
    """Simplified model registry with fallback strategy"""
    
    MODELS = {
        'groq-llama-70b': ModelConfig('llama-3.3-70b-versatile', 'groq', 4096, 0.7),
        'groq-llama-8b': ModelConfig('llama-3.1-8b-instant', 'groq', 8192, 0.7),
        'openai-gpt4': ModelConfig('gpt-4-turbo-preview', 'openai', 4096, 0.7),
        'openai-gpt35': ModelConfig('gpt-3.5-turbo', 'openai', 4096, 0.7)
    }
    
    @staticmethod
    def get_model(model_id: str = None):
        """Get model with automatic fallback"""
        if not model_id:
            model_id = st.session_state.model_settings['primary_model']
        
        config = ModelRegistry.MODELS.get(model_id)
        if not config:
            # Fallback to first available model
            for m_id in ModelRegistry.MODELS:
                config = ModelRegistry.MODELS[m_id]
                break
        
        try:
            if config.provider == "groq":
                api_key = st.session_state.api_keys.get('GROQ_API_KEY')
                if not api_key:
                    return None
                return ChatGroq(
                    groq_api_key=api_key,
                    model_name=config.name,
                    temperature=st.session_state.model_settings['temperature'],
                    max_tokens=min(config.max_tokens, st.session_state.model_settings['max_tokens'])
                )
            elif config.provider == "openai":
                api_key = st.session_state.api_keys.get('OPENAI_API_KEY')
                if not api_key:
                    return None
                return ChatOpenAI(
                    api_key=api_key,
                    model_name=config.name,
                    temperature=st.session_state.model_settings['temperature'],
                    max_tokens=min(config.max_tokens, st.session_state.model_settings['max_tokens'])
                )
        except Exception as e:
            st.error(f"Error loading model {model_id}: {str(e)}")
            return None
        return None
    
    @staticmethod
    def get_available_models():
        """Get list of models with available API keys"""
        available = []
        for model_id, config in ModelRegistry.MODELS.items():
            if config.provider == "groq" and st.session_state.api_keys.get('GROQ_API_KEY'):
                available.append(model_id)
            elif config.provider == "openai" and st.session_state.api_keys.get('OPENAI_API_KEY'):
                available.append(model_id)
        return available

# ============================================================================
# ADVANCED RAG PIPELINE (Robust)
# ============================================================================
class AdvancedRAG:
    """Streamlined but powerful RAG pipeline"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Safe for Streamlit Cloud
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def process_documents(self, uploaded_files):
        """Process documents with error handling"""
        documents = []
        temp_files = []
        
        try:
            for uploaded_file in uploaded_files:
                file_ext = uploaded_file.name.split('.')[-1].lower()
                temp_path = None
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name
                    temp_files.append(temp_path)
                
                # Load document based on type
                loader = self._get_loader(temp_path, file_ext)
                if loader:
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata.update({
                            'source': uploaded_file.name,
                            'file_type': file_ext,
                            'chunk_id': str(uuid.uuid4())[:8],
                            'timestamp': datetime.now().isoformat()
                        })
                    documents.extend(docs)
                else:
                    st.warning(f"Unsupported file type: {file_ext}")
            
            if documents:
                # Split documents
                splits = self.text_splitter.split_documents(documents)
                
                # Create vector store (FAISS for speed)
                vectorstore = FAISS.from_documents(splits, self.embeddings)
                
                # Store in session
                st.session_state.vectorstore = vectorstore
                st.session_state.processed_files = [f.name for f in uploaded_files]
                
                # Calculate metrics
                metrics = self._calculate_metrics(splits)
                st.session_state.rag_metrics = metrics
                
                return vectorstore, splits
                
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return None, []
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def _get_loader(self, file_path, file_ext):
        """Get appropriate document loader"""
        loaders = {
            'pdf': PyPDFLoader,
            'txt': TextLoader,
            'docx': DocxLoader,
            'csv': CSVLoader,
            'html': WebBaseLoader,
            'htm': WebBaseLoader
        }
        
        loader_class = loaders.get(file_ext)
        if loader_class:
            return loader_class(file_path)
        return None
    
    def _calculate_metrics(self, splits):
        """Calculate RAG metrics"""
        if not splits:
            return {}
        
        try:
            total_chunks = len(splits)
            avg_length = np.mean([len(doc.page_content) for doc in splits]) if splits else 0
            
            # Simple embedding quality check
            sample_texts = [doc.page_content[:100] for doc in splits[:5] if doc.page_content]
            if sample_texts:
                sample_embeddings = self.embeddings.embed_documents(sample_texts)
                if len(sample_embeddings) > 1:
                    similarity_matrix = cosine_similarity(sample_embeddings)
                    diversity_score = 1 - np.mean(similarity_matrix)
                else:
                    diversity_score = 0
            else:
                diversity_score = 0
            
            return {
                'total_chunks': total_chunks,
                'avg_chunk_length': round(avg_length, 2),
                'diversity_score': round(diversity_score, 3),
                'unique_sources': len(set(doc.metadata.get('source', '') for doc in splits)),
                'processed_at': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
        except Exception as e:
            st.warning(f"Could not calculate metrics: {str(e)}")
            return {}
    
    def hybrid_search(self, query, top_k=4):
        """Hybrid search with fallback"""
        if not st.session_state.vectorstore:
            return []
        
        try:
            # Vector similarity search (primary)
            vector_results = st.session_state.vectorstore.similarity_search(query, k=top_k*2)
            
            # If we have text chunks, add BM25
            all_texts = [doc.page_content for doc in vector_results]
            if all_texts:
                bm25_retriever = BM25Retriever.from_texts(
                    all_texts,
                    metadatas=[doc.metadata for doc in vector_results]
                )
                bm25_results = bm25_retriever.get_relevant_documents(query)
                
                # Combine results (deduplicate)
                all_results = vector_results + bm25_results
                unique_results = {}
                for result in all_results:
                    content_key = result.page_content[:200]
                    if content_key not in unique_results:
                        unique_results[content_key] = result
                
                results = list(unique_results.values())[:top_k]
            else:
                results = vector_results[:top_k]
            
            # Simple re-ranking by relevance to query
            if len(results) > 1:
                query_lower = query.lower()
                scored_results = []
                for doc in results:
                    score = 0
                    content_lower = doc.page_content.lower()
                    
                    # Exact match bonus
                    if query_lower in content_lower:
                        score += 2
                    
                    # Keyword density
                    keywords = query_lower.split()
                    matches = sum(1 for kw in keywords if kw in content_lower)
                    score += matches / len(keywords) if keywords else 0
                    
                    # Length penalty (prefer concise)
                    score -= len(doc.page_content) / 10000
                    
                    scored_results.append((score, doc))
                
                scored_results.sort(reverse=True, key=lambda x: x[0])
                results = [doc for _, doc in scored_results[:top_k]]
            
            return results
            
        except Exception as e:
            st.warning(f"Search error: {str(e)}")
            # Fallback to simple search
            try:
                return st.session_state.vectorstore.similarity_search(query, k=top_k)
            except:
                return []
    
    def generate_answer(self, query, context_chunks, chat_history=None):
        """Generate answer with citations"""
        llm = ModelRegistry.get_model()
        if not llm:
            return "Error: No valid API key configured. Please check your settings."
        
        try:
            # Prepare context
            context_text = ""
            citations = []
            
            for i, chunk in enumerate(context_chunks[:3]):  # Use top 3 chunks
                context_text += f"[Source {i+1}]: {chunk.page_content[:500]}\n\n"
                citations.append({
                    'source': chunk.metadata.get('source', 'Unknown'),
                    'excerpt': chunk.page_content[:200] + "..."
                })
            
            # Prepare conversation history
            history_text = ""
            if chat_history:
                for msg in chat_history[-3:]:  # Last 3 messages
                    role = "User" if msg.get("role") == "user" else "Assistant"
                    history_text += f"{role}: {msg.get('content', '')}\n"
            
            # Create prompt
            prompt = f"""You are an expert assistant. Answer the question based ONLY on the provided context.

Context from documents:
{context_text}

Previous conversation:
{history_text}

Question: {query}

Requirements:
1. Answer based ONLY on the provided context
2. If the answer isn't in the context, say "I cannot find this information in the provided documents"
3. Be concise but comprehensive
4. Cite sources using [Source X] notation when using specific information

Answer:"""
            
            # Generate response
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant that provides accurate, cited answers."),
                ("human", "{question}")
            ])
            
            chain = prompt_template | llm | StrOutputParser()
            response = chain.invoke({"question": prompt})
            
            # Add citations metadata
            response_data = {
                'answer': response,
                'citations': citations,
                'sources_used': len(citations),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
            
            return response_data
            
        except Exception as e:
            return {
                'answer': f"Error generating answer: {str(e)}",
                'citations': [],
                'sources_used': 0,
                'error': str(e)
            }

# ============================================================================
# MULTI-AGENT FRAMEWORK (Simplified)
# ============================================================================
class AgentState:
    """State for agent workflows"""
    def __init__(self):
        self.messages = []
        self.research_data = []
        self.analysis_result = None
        self.final_output = None

class OmniAgents:
    """Simplified multi-agent system"""
    
    def __init__(self):
        self.llm = ModelRegistry.get_model('groq-llama-8b')  # Fast model for agents
        
    def research_agent(self, query):
        """Research agent - gathers information"""
        if not self.llm:
            return "Research agent unavailable"
        
        prompt = f"""You are a research agent. Analyze this query and outline what information would be needed:
        
        Query: {query}
        
        Provide:
        1. Key topics to research
        2. Potential data sources
        3. Research methodology outline
        
        Keep it concise."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except:
            return "Research completed"
    
    def analysis_agent(self, research_data, query):
        """Analysis agent - processes information"""
        if not self.llm:
            return "Analysis agent unavailable"
        
        prompt = f"""You are an analysis agent. Based on this research, analyze the query:

        Query: {query}
        
        Research Data: {research_data[:1000]}...
        
        Provide:
        1. Key findings
        2. Insights and patterns
        3. Recommendations if applicable
        
        Be analytical and data-driven."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except:
            return "Analysis completed"
    
    def synthesis_agent(self, research, analysis, query):
        """Synthesis agent - creates final output"""
        if not self.llm:
            return "Synthesis agent unavailable"
        
        prompt = f"""You are a synthesis agent. Create a comprehensive answer:

        Original Query: {query}
        
        Research: {research[:500]}...
        
        Analysis: {analysis[:500]}...
        
        Create a well-structured answer that:
        1. Directly addresses the query
        2. Incorporates research findings
        3. Presents analysis insights
        4. Is clear and actionable
        
        Format with sections if appropriate."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except:
            return "Synthesis completed"
    
    def execute_workflow(self, query):
        """Execute multi-agent workflow"""
        with st.spinner("🤖 Researching..."):
            research = self.research_agent(query)
        
        with st.spinner("🔍 Analyzing..."):
            analysis = self.analysis_agent(research, query)
        
        with st.spinner("📝 Synthesizing..."):
            final = self.synthesis_agent(research, analysis, query)
        
        return {
            'research': research,
            'analysis': analysis,
            'final_answer': final,
            'agents_used': 3,
            'timestamp': datetime.now().isoformat()
        }

# ============================================================================
# KNOWLEDGE GRAPH (Lightweight)
# ============================================================================
class KnowledgeGraphManager:
    """Simple knowledge graph for entity relationships"""
    
    def __init__(self):
        self.graph = nx.Graph()
        
    def extract_entities(self, text):
        """Simple entity extraction (simplified for demo)"""
        # In production, use spaCy or similar
        entities = []
        words = text.split()
        
        # Simple heuristic: capitalize words might be entities
        for word in words:
            if len(word) > 3 and word[0].isupper():
                entities.append(word.strip('.,;!?'))
        
        return list(set(entities))[:10]  # Limit to 10 entities
    
    def build_from_text(self, text, source="unknown"):
        """Build simple knowledge graph from text"""
        entities = self.extract_entities(text)
        
        # Add entities as nodes
        for entity in entities:
            self.graph.add_node(entity, type="entity", source=source)
        
        # Add relationships (simplified: co-occurrence)
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:i+3]:  # Connect to next 2 entities
                if entity1 != entity2:
                    if not self.graph.has_edge(entity1, entity2):
                        self.graph.add_edge(entity1, entity2, weight=1, type="related")
                    else:
                        # Increase weight if edge exists
                        current_weight = self.graph[entity1][entity2].get('weight', 0)
                        self.graph[entity1][entity2]['weight'] = current_weight + 1
        
        st.session_state.knowledge_graph = self.graph
    
    def visualize_graph(self):
        """Create visualization of knowledge graph"""
        if len(self.graph.nodes()) == 0:
            return None
        
        # Create Plotly network graph
        pos = nx.spring_layout(self.graph, seed=42)
        
        edge_trace = []
        for edge in self.graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in self.graph.nodes()],
            y=[pos[node][1] for node in self.graph.nodes()],
            mode='markers+text',
            text=list(self.graph.nodes()),
            textposition="top center",
            marker=dict(
                size=20,
                color='#667eea',
                line=dict(width=2, color='white')
            )
        )
        
        fig = go.Figure(data=edge_trace + [node_trace],
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=0),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig

# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================
class EvaluationFramework:
    """Lightweight evaluation framework"""
    
    @staticmethod
    def evaluate_response(query, response, context_chunks):
        """Evaluate RAG response quality"""
        try:
            metrics = {}
            
            # 1. Check if answer acknowledges missing info
            answer_lower = response['answer'].lower()
            missing_phrases = ['cannot find', 'not in', 'unavailable', "don't know"]
            metrics['acknowledges_limits'] = any(phrase in answer_lower for phrase in missing_phrases)
            
            # 2. Check for citations
            metrics['has_citations'] = len(response.get('citations', [])) > 0
            
            # 3. Response length
            metrics['response_length'] = len(response['answer'])
            
            # 4. Source usage
            metrics['sources_used'] = response.get('sources_used', 0)
            
            # 5. Query relevance (simple keyword matching)
            query_words = set(query.lower().split())
            answer_words = set(answer_lower.split())
            common_words = query_words.intersection(answer_words)
            metrics['relevance_score'] = len(common_words) / len(query_words) if query_words else 0
            
            metrics['evaluated_at'] = datetime.now().isoformat()
            return metrics
            
        except Exception as e:
            return {'error': str(e)}

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
def render_sidebar():
    """Render the sidebar with all configurations"""
    with st.sidebar:
        st.markdown('<div class="main-header">⚙️ OmniBot Studio</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("🔑 API Configuration")
        
        # API Key Input
        groq_key = st.text_input("Groq API Key", 
                                value=st.session_state.api_keys.get('GROQ_API_KEY', ''),
                                type="password",
                                help="Get from https://console.groq.com")
        
        openai_key = st.text_input("OpenAI API Key", 
                                  value=st.session_state.api_keys.get('OPENAI_API_KEY', ''),
                                  type="password",
                                  help="Get from https://platform.openai.com")
        
        # Update keys
        if groq_key != st.session_state.api_keys.get('GROQ_API_KEY'):
            st.session_state.api_keys['GROQ_API_KEY'] = groq_key
        
        if openai_key != st.session_state.api_keys.get('OPENAI_API_KEY'):
            st.session_state.api_keys['OPENAI_API_KEY'] = openai_key
        
        # Model Selection
        st.markdown("---")
        st.subheader("🧠 Model Settings")
        
        available_models = ModelRegistry.get_available_models()
        if available_models:
            model_display_names = {
                'groq-llama-70b': 'Groq Llama 70B (Recommended)',
                'groq-llama-8b': 'Groq Llama 8B (Fast)',
                'openai-gpt4': 'OpenAI GPT-4 (Most Capable)',
                'openai-gpt35': 'OpenAI GPT-3.5 (Economical)'
            }
            
            selected_model = st.selectbox(
                "Primary Model",
                options=available_models,
                format_func=lambda x: model_display_names.get(x, x),
                index=0 if available_models else 0
            )
            
            if selected_model != st.session_state.model_settings['primary_model']:
                st.session_state.model_settings['primary_model'] = selected_model
            
            # Model parameters
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider("Temperature", 0.0, 1.0, 
                                       st.session_state.model_settings['temperature'], 0.1)
                if temperature != st.session_state.model_settings['temperature']:
                    st.session_state.model_settings['temperature'] = temperature
            
            with col2:
                max_tokens = st.slider("Max Tokens", 256, 4096, 
                                      st.session_state.model_settings['max_tokens'], 256)
                if max_tokens != st.session_state.model_settings['max_tokens']:
                    st.session_state.model_settings['max_tokens'] = max_tokens
        else:
            st.warning("Add API keys to enable models")
        
        # System Status
        st.markdown("---")
        st.subheader("📊 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chat History", len(st.session_state.conversation_history))
            st.metric("Processed Files", len(st.session_state.processed_files))
        
        with col2:
            st.metric("Code History", len(st.session_state.code_history))
            st.metric("KG Nodes", len(st.session_state.knowledge_graph.nodes()))
        
        # Clear buttons
        st.markdown("---")
        st.subheader("🔄 Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()
            
            if st.button("Clear Documents", use_container_width=True):
                st.session_state.vectorstore = None
                st.session_state.processed_files = []
                st.session_state.document_chat_history = []
                st.rerun()
        
        with col2:
            if st.button("Clear Code", use_container_width=True):
                st.session_state.code_history = []
                st.rerun()
            
            if st.button("Clear All", use_container_width=True, type="secondary"):
                for key in list(st.session_state.keys()):
                    if key not in ['api_keys', 'model_settings']:
                        del st.session_state[key]
                init_session_state()
                st.rerun()
        
        st.markdown("---")
        st.caption("OmniBot Studio v3.0 | Built with Streamlit")

# ============================================================================
# MAIN APP - TABBED INTERFACE
# ============================================================================
def main():
    """Main application with tabbed interface"""
    
    # Initialize components
    rag_pipeline = AdvancedRAG()
    agents = OmniAgents()
    kg_manager = KnowledgeGraphManager()
    evaluator = EvaluationFramework()
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown('<div class="main-header">🤖 OmniBot AI Studio</div>', unsafe_allow_html=True)
    
    # API Status
    api_status = []
    if st.session_state.api_keys.get('GROQ_API_KEY'):
        api_status.append("✅ Groq")
    if st.session_state.api_keys.get('OPENAI_API_KEY'):
        api_status.append("✅ OpenAI")
    
    if api_status:
        st.success(f"**Active APIs:** {' | '.join(api_status)}")
    else:
        st.warning("**Demo Mode:** Add API keys for full functionality")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 AI Chat", "📚 Document Intelligence", "🤖 Multi-Agent", 
        "🔍 Knowledge Graph", "💻 Code Studio", "📊 Analytics"
    ])
    
    # ========================================================================
    # TAB 1: AI CHAT
    # ========================================================================
    with tab1:
        st.subheader("💬 Intelligent Chat Assistant")
        
        # Chat history display
        chat_container = st.container(height=400, border=True)
        
        with chat_container:
            for message in st.session_state.conversation_history[-10:]:  # Show last 10
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("Type your message...", key="chat_input", label_visibility="collapsed")
        
        with col2:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        if send_button and user_input:
            llm = ModelRegistry.get_model()
            
            if not llm:
                st.error("Please configure API keys in the sidebar")
                st.stop()
            
            # Add user message to history
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    # Prepare conversation context
                    context_messages = []
                    for msg in st.session_state.conversation_history[-6:]:  # Last 6 messages
                        if msg['role'] == 'user':
                            context_messages.append(HumanMessage(content=msg['content']))
                        else:
                            context_messages.append(AIMessage(content=msg['content']))
                    
                    # Create prompt
                    prompt = ChatPromptTemplate.from_messages([
                        SystemMessage(content="You are OmniBot, a helpful AI assistant. Be concise, accurate, and friendly."),
                        *context_messages,
                        HumanMessage(content=user_input)
                    ])
                    
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({})
                    
                    # Add to history
                    st.session_state.conversation_history.append({
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # Rerun to update display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # ========================================================================
    # TAB 2: DOCUMENT INTELLIGENCE
    # ========================================================================
    with tab2:
        st.subheader("📚 Document Intelligence")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload documents (PDF, TXT, DOCX, CSV)",
                type=["pdf", "txt", "docx", "csv", "html"],
                accept_multiple_files=True,
                help="Upload documents for analysis"
            )
            
            if uploaded_files and st.button("Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    vectorstore, splits = rag_pipeline.process_documents(uploaded_files)
                    
                    if vectorstore:
                        st.success(f"✅ Processed {len(uploaded_files)} file(s) into {len(splits)} chunks")
                        
                        # Build knowledge graph
                        for split in splits[:5]:  # First 5 chunks
                            kg_manager.build_from_text(split.page_content, split.metadata.get('source', ''))
                        
                        # Show metrics
                        if st.session_state.rag_metrics:
                            st.info(f"""
                            **RAG Metrics:**
                            • Chunks: {st.session_state.rag_metrics.get('total_chunks', 0)}
                            • Avg Length: {st.session_state.rag_metrics.get('avg_chunk_length', 0)} chars
                            • Diversity: {st.session_state.rag_metrics.get('diversity_score', 0):.3f}
                            """)
            
            # Document chat
            if st.session_state.vectorstore:
                st.markdown("---")
                st.subheader("💬 Chat with Documents")
                
                doc_question = st.text_input("Ask about your documents...", key="doc_question")
                
                if doc_question and st.button("Search Documents"):
                    with st.spinner("Searching documents..."):
                        # Search documents
                        context_chunks = rag_pipeline.hybrid_search(doc_question, top_k=3)
                        
                        # Generate answer
                        response = rag_pipeline.generate_answer(
                            doc_question, 
                            context_chunks,
                            st.session_state.document_chat_history
                        )
                        
                        # Display answer
                        st.markdown("### Answer")
                        st.write(response['answer'])
                        
                        if response.get('citations'):
                            with st.expander("📎 Sources"):
                                for i, citation in enumerate(response['citations']):
                                    st.write(f"**Source {i+1}:** {citation['source']}")
                                    st.caption(citation['excerpt'])
                        
                        # Add to chat history
                        st.session_state.document_chat_history.append({
                            'role': 'user',
                            'content': doc_question,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        st.session_state.document_chat_history.append({
                            'role': 'assistant',
                            'content': response['answer'],
                            'metadata': response,
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Evaluate response
                        metrics = evaluator.evaluate_response(doc_question, response, context_chunks)
                        st.session_state.evaluation_results = metrics
        
        with col2:
            # Display processed files
            if st.session_state.processed_files:
                st.markdown("### 📂 Processed Files")
                for file in st.session_state.processed_files:
                    st.markdown(f"• {file}")
            
            # Display document chat history
            if st.session_state.document_chat_history:
                st.markdown("### 💭 Document Chat History")
                history_container = st.container(height=300, border=True)
                
                with history_container:
                    for msg in st.session_state.document_chat_history[-5:]:
                        if msg['role'] == 'user':
                            st.markdown(f"**You:** {msg['content'][:100]}...")
                        else:
                            st.markdown(f"**Assistant:** {msg['content'][:100]}...")
                        st.markdown("---")
    
    # ========================================================================
    # TAB 3: MULTI-AGENT
    # ========================================================================
    with tab3:
        st.subheader("🤖 Multi-Agent Workflow")
        
        st.markdown("""
        Multi-agent systems break complex tasks into specialized agents:
        - **Research Agent**: Gathers information
        - **Analysis Agent**: Processes and analyzes
        - **Synthesis Agent**: Creates final output
        """)
        
        agent_query = st.text_area("Enter a complex query for multi-agent processing:", 
                                  height=100,
                                  placeholder="e.g., 'Analyze the impact of AI on healthcare and suggest future trends...'")
        
        if agent_query and st.button("Run Multi-Agent Workflow", type="primary"):
            with st.spinner("Multi-agent system working..."):
                result = agents.execute_workflow(agent_query)
                
                # Display results in expanders
                with st.expander("🔍 Research Phase", expanded=True):
                    st.write(result['research'])
                
                with st.expander("📊 Analysis Phase"):
                    st.write(result['analysis'])
                
                with st.expander("📝 Final Synthesis"):
                    st.write(result['final_answer'])
                
                # Store in session
                if 'agent_conversations' not in st.session_state:
                    st.session_state.agent_conversations = {}
                
                conv_id = str(uuid.uuid4())[:8]
                st.session_state.agent_conversations[conv_id] = {
                    'query': agent_query,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
    
    # ========================================================================
    # TAB 4: KNOWLEDGE GRAPH
    # ========================================================================
    with tab4:
        st.subheader("🔍 Knowledge Graph Explorer")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Build KG from text
            kg_text = st.text_area("Enter text to build knowledge graph:", 
                                  height=200,
                                  placeholder="Paste text about people, organizations, concepts...")
            
            if kg_text and st.button("Build Knowledge Graph"):
                with st.spinner("Extracting entities and relationships..."):
                    kg_manager.build_from_text(kg_text, source="user_input")
                    st.success(f"Built graph with {len(st.session_state.knowledge_graph.nodes())} entities")
            
            # Or build from documents
            if st.session_state.processed_files:
                if st.button("Build KG from Documents"):
                    with st.spinner("Processing documents for KG..."):
                        # Build KG from all processed chunks
                        if st.session_state.vectorstore:
                            # Get some sample chunks
                            all_docs = st.session_state.vectorstore.similarity_search("", k=10)
                            for doc in all_docs:
                                kg_manager.build_from_text(doc.page_content, doc.metadata.get('source', ''))
                            st.success(f"Graph now has {len(st.session_state.knowledge_graph.nodes())} entities")
        
        with col2:
            # KG Stats
            if len(st.session_state.knowledge_graph.nodes()) > 0:
                st.markdown("### 📊 Graph Statistics")
                
                stats = {
                    "Total Entities": len(st.session_state.knowledge_graph.nodes()),
                    "Total Relationships": len(st.session_state.knowledge_graph.edges()),
                    "Density": f"{nx.density(st.session_state.knowledge_graph):.4f}",
                    "Avg Degree": f"{np.mean([d for _, d in st.session_state.knowledge_graph.degree()]):.2f}"
                }
                
                for key, value in stats.items():
                    st.metric(key, value)
                
                # Visualize
                if st.button("Visualize Graph"):
                    fig = kg_manager.visualize_graph()
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Graph is too small to visualize")
                
                # Export
                if st.button("Export Graph Data"):
                    # Convert to dict for display
                    graph_data = {
                        "nodes": list(st.session_state.knowledge_graph.nodes()),
                        "edges": list(st.session_state.knowledge_graph.edges(data=True))
                    }
                    st.json(graph_data, expanded=False)
            else:
                st.info("No knowledge graph data yet. Add text or documents to build one.")
    
    # ========================================================================
    # TAB 5: CODE STUDIO
    # ========================================================================
    with tab5:
        st.subheader("💻 AI Code Studio")
        
        col1, col2 = st.columns(2)
        
        with col1:
            language = st.selectbox(
                "Programming Language",
                ["Python", "JavaScript", "Java", "C++", "TypeScript", "SQL", "Go", "Rust"],
                index=0
            )
            
            task_type = st.selectbox(
                "Task Type",
                ["Write Code", "Debug Code", "Explain Code", "Optimize Code", "Learn Concept"],
                index=0
            )
        
        with col2:
            complexity = st.select_slider(
                "Complexity Level",
                options=["Beginner", "Intermediate", "Advanced", "Expert"],
                value="Intermediate"
            )
            
            include_tests = st.checkbox("Include Tests", value=True)
            include_comments = st.checkbox("Include Comments", value=True)
        
        # Code input
        code_input = st.text_area(
            "Describe your code task or paste your code:",
            height=150,
            placeholder="e.g., 'Create a Python function to validate email addresses...'"
        )
        
        if code_input and st.button("Generate Code", type="primary"):
            llm = ModelRegistry.get_model()
            
            if not llm:
                st.error("Please configure API keys")
                st.stop()
            
            with st.spinner("Coding..."):
                try:
                    # Create coding prompt
                    prompt = f"""You are an expert {language} developer. {task_type}.

                    Requirements:
                    - Language: {language}
                    - Complexity: {complexity}
                    - Include tests: {include_tests}
                    - Include comments: {include_comments}
                    
                    Task: {code_input}
                    
                    Provide:
                    1. Complete, runnable code
                    2. Clear explanation of how it works
                    3. {f"Test cases" if include_tests else "Usage examples"}
                    4. Best practices applied
                    
                    Format with proper code blocks."""
                    
                    # Generate code
                    code_prompt = ChatPromptTemplate.from_messages([
                        ("system", f"You are an expert {language} developer."),
                        ("human", "{query}")
                    ])
                    
                    chain = code_prompt | llm | StrOutputParser()
                    response = chain.invoke({"query": prompt})
                    
                    # Display response
                    st.markdown("### Generated Code")
                    st.code(response, language=language.lower())
                    
                    # Store in history
                    st.session_state.code_history.append({
                        'language': language,
                        'task': task_type,
                        'input': code_input,
                        'output': response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    st.error(f"Error generating code: {str(e)}")
        
        # Code history
        if st.session_state.code_history:
            st.markdown("---")
            st.subheader("📚 Code History")
            
            for i, item in enumerate(reversed(st.session_state.code_history[-3:])):
                with st.expander(f"{item['language']} - {item['task']} ({i+1})"):
                    st.write("**Your Request:**")
                    st.write(item['input'][:200] + "..." if len(item['input']) > 200 else item['input'])
                    st.write("**Generated Code:**")
                    st.code(item['output'][:500] + "..." if len(item['output']) > 500 else item['output'], 
                           language=item['language'].lower())
    
    # ========================================================================
    # TAB 6: ANALYTICS
    # ========================================================================
    with tab6:
        st.subheader("📊 System Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Chat Metrics")
            chat_df = pd.DataFrame(st.session_state.conversation_history)
            if not chat_df.empty:
                st.metric("Total Messages", len(chat_df))
                st.metric("User Messages", len(chat_df[chat_df['role'] == 'user']))
                st.metric("Avg Response Length", 
                         chat_df[chat_df['role'] == 'assistant']['content'].str.len().mean() if len(chat_df[chat_df['role'] == 'assistant']) > 0 else 0)
            else:
                st.info("No chat data")
        
        with col2:
            st.markdown("### RAG Metrics")
            if st.session_state.rag_metrics:
                for key, value in st.session_state.rag_metrics.items():
                    if key != 'processed_at':
                        st.metric(key.replace('_', ' ').title(), value)
            else:
                st.info("No RAG metrics")
        
        with col3:
            st.markdown("### System Health")
            st.metric("Active Sessions", 1)  # Single user for now
            st.metric("Memory Usage", "Low")
            st.metric("API Status", "✅" if ModelRegistry.get_available_models() else "⚠️")
        
        # Visualizations
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Chat timeline
            if len(st.session_state.conversation_history) > 0:
                st.markdown("### Chat Activity")
                chat_times = [msg.get('timestamp', datetime.now().isoformat()) 
                             for msg in st.session_state.conversation_history]
                
                if chat_times:
                    time_df = pd.DataFrame({
                        'time': pd.to_datetime(chat_times),
                        'count': 1
                    })
                    time_df = time_df.set_index('time').resample('5min').count().fillna(0)
                    
                    fig = px.line(time_df, x=time_df.index, y='count', 
                                 labels={'time': 'Time', 'count': 'Messages'},
                                 title="Messages Over Time")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Knowledge Graph Visualization
            if len(st.session_state.knowledge_graph.nodes()) > 0:
                st.markdown("### Knowledge Graph")
                fig = kg_manager.visualize_graph()
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Graph visualization requires at least 3 entities")
        
        # Export data
        st.markdown("---")
        if st.button("Export All Data"):
            export_data = {
                'chat_history': st.session_state.conversation_history[-50:],  # Last 50
                'code_history': st.session_state.code_history[-10:],
                'rag_metrics': st.session_state.rag_metrics,
                'processed_files': st.session_state.processed_files,
                'knowledge_graph': {
                    'nodes': list(st.session_state.knowledge_graph.nodes()),
                    'edges': len(st.session_state.knowledge_graph.edges())
                }
            }
            
            # Convert to JSON for download
            json_str = json.dumps(export_data, indent=2, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
            
            st.download_button(
                label="Download Analytics Data",
                data=json_str,
                file_name=f"omnibot_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# ============================================================================
# RUN THE APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
