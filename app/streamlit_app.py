import streamlit as st
import sys
import os
import pandas as pd
from datetime import datetime

# --- DEVELOPMENT FLAG ---
DEBUG_MODE = False  # For dev testing only

# --- Setup project imports ---
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(os.path.join(project_root, 'src'))
sys.path.insert(0, project_root)

# Imports from src/
try:
    from data_preprocessing import DataPreprocessor
    from chatbot import HybridEmpowermentChatbot
except ImportError as e:
    st.error(f"Failed to import core modules. Ensure 'src' folder is set up correctly. Error: {e}")
    st.stop()

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Women's Empowerment Assistant",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "EmpowerHer Chatbot - AI-powered women's empowerment assistant"
    }
)

# Add loading screen
@st.cache_data
def show_loading_screen():
    """Show loading screen while app initializes"""
    with st.spinner("🚀 Loading EmpowerHer Chatbot..."):
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>💫 Women's Empowerment AI Companion</h2>
            <p>Initializing your supportive AI assistant...</p>
            <div style="margin: 1rem 0;">
                <div style="display: inline-block; animation: spin 2s linear infinite;">
                    🌸
                </div>
            </div>
        </div>
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """, unsafe_allow_html=True)
        import time
        time.sleep(0.5)  # Brief pause to show loading

# --- Custom CSS ---
st.markdown("""
<style>
.welcome-banner {
    padding: 20px;
    background-color: #f0e6ff;
    border-radius: 10px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stButton>button {
    background-color: #8a2be2;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
@st.cache_data(show_spinner="Pre-processing dataset...")
def load_data():
    """Load and preprocess dataset"""
    try:
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data(os.path.join(project_root, 'dataset/womenchatbot.csv'))
        return preprocessor.preprocess_dataset(df)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource(show_spinner="Initializing AI system...")
def initialize_chatbot():
    """Initialize chatbot with caching"""
    try:
        df = load_data()
        if df is None:
            return None
            
        # Check for fine-tuned model in multiple possible locations
        possible_paths = [
            os.path.join(project_root, "src/fine_tuned_model"),
            os.path.join(project_root, "models/fine_tuned_model"),
            os.path.join(project_root, "models/best_model")
        ]
        
        model_path_init = None
        for path in possible_paths:
            if os.path.exists(path) and os.listdir(path):
                # Check if it has the required model files
                has_config = os.path.exists(os.path.join(path, 'config.json'))
                has_model = (os.path.exists(os.path.join(path, 'pytorch_model.bin')) or 
                           os.path.exists(os.path.join(path, 'model.safetensors')))
                if has_config and has_model:
                    model_path_init = path
                    break
        
        chatbot = HybridEmpowermentChatbot(
            tfidf_dataframe=df,
            fine_tuned_model_path=model_path_init
        )
        
        return chatbot, model_path_init
    except Exception as e:
        st.error(f"Error initializing chatbot: {e}")
        return None, None

def create_welcome_banner():
    st.markdown("""
    <div class="welcome-banner">
        <h1 style="margin:0; font-size:2.5rem;">💫 Women's Empowerment AI Companion</h1>
    </div>
    """, unsafe_allow_html=True)

# --- MAIN APPLICATION FUNCTION ---
def main():
    create_welcome_banner()

    # --- Initialize Chatbot ---
    if "hybrid_chatbot" not in st.session_state:
        chatbot, model_path = initialize_chatbot()
        if chatbot is not None:
            st.session_state.hybrid_chatbot = chatbot
            if model_path:
                st.info(f"🤖 Found fine-tuned model at: {model_path}")
            else:
                st.info("🤖 Using TF-IDF knowledge base only (no fine-tuned model found)")
            st.success("🎉 System ready! You can now chat.")
        else:
            st.error("❌ System initialization failed")
            st.stop()

    # --- Messages State ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "HELLO Keza welcome to your Assistant :🦋 Hello amazing woman! I'm your virtual support system, ready to help you spread your wings.",
                "category": "creative_welcome",
                "timestamp": datetime.now().strftime("%H:%M")
            }
        ]

    # --- Layout ---
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### AI System")
        if st.session_state.hybrid_chatbot.fine_tuned_chatbot:
            st.success("**Advanced AI**: ✅ Online")
        else:
            st.info("**Standard Mode**: ✅ Online (Knowledge Base Only)")

        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "HELLO Keza welcome to your Assistant :🦋 Hello amazing woman! I'm your virtual support system, ready to help you spread your wings.",
                    "category": "creative_welcome",
                    "timestamp": datetime.now().strftime("%H:%M")
                }
            ]
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 🎯 Quick Inspiration")
        quick_topics = {
            "Boost my confidence today": "I'd love to help boost your confidence!",
            "Career growth strategies": "Career growth is such an exciting journey!",
            "Work-life balance magic": "Finding that sweet spot between work and life is magical!",
        }

        for topic in quick_topics.keys():
            if st.button(f"💬 {topic}", key=topic, use_container_width=True):
                st.session_state.messages.append({
                    "role": "user",
                    "content": topic,
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                with st.spinner("✨ AI is thinking..."):
                    result = st.session_state.hybrid_chatbot.get_response(topic)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['response'],
                    "source": result.get('source', 'tfidf'),
                    "timestamp": datetime.now().strftime("%H:%M")
                })
                st.rerun()

    # --- Chat Display in col1 ---
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                content = message["content"]
                if message.get('category') not in ('creative_welcome', 'context_aware'):
                    source = message.get('source')
                    if source == 'fine_tuned':
                        content += '\n\n*🚀 Advanced AI Response*'
                    elif source == 'tfidf':
                        content += '\n\n*💡 Knowledge Base Response*'
                st.markdown(content)

    # --- Chat Input (must be outside columns) ---
    user_input = st.chat_input("💫 Share your dreams, questions, or journey...")
    if user_input:
        # Prevent duplicate processing - check if last message is the same
        should_process = True
        if len(st.session_state.messages) >= 2:
            last_user_msg = [m for m in st.session_state.messages if m["role"] == "user"]
            if last_user_msg and last_user_msg[-1]["content"] == user_input:
                should_process = False
        
        if should_process:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            # Get bot response
            with st.spinner("✨ AI is thinking..."):
                result = st.session_state.hybrid_chatbot.get_response(user_input)
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['response'],
                "source": result.get('source', 'tfidf'),
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.info("🔄 Please refresh the page or try again in a moment.")
        st.stop()
