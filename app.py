import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="🎙️ Persona Interview Simulator",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat-like styling
st.markdown("""
<style>
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }
    .chat-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px 10px 0 0;
        margin: -20px -20px 20px -20px;
    }
    .chat-messages {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        background: white;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        text-align: right;
        max-width: 70%;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f8f9fa;
        color: #212529;
        padding: 10px 15px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        text-align: left;
        max-width: 70%;
        border: 1px solid #dee2e6;
    }
    .input-container {
        background: white;
        padding: 15px;
        border-radius: 0 0 10px 10px;
        border-top: 1px solid #dee2e6;
    }
    .stButton > button {
        border-radius: 20px;
        height: 40px;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        height: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Check if required environment variables are set
required_vars = ["OPENROUTER_API_KEY", "PINECONE_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    st.info("Please create a `.env` file with the following variables:")
    st.code(f"""
OPENROUTER_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # optional
    """)
    st.stop()

# Try to import and set up the QA chain
try:
    from pilot import qa_chain, profile
    
    # Chat container with header
    st.markdown(f"""
    <div class="chat-container">
        <div class="chat-header">
            <h2 style="margin: 0; font-size: 1.5rem;">🎙️ {profile.get('name', 'Unknown')}</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 0.9rem;">{profile.get('bio', 'AI Interview Simulator')}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Persona details in a small info box
    with st.expander("ℹ️ About this persona", expanded=False):
        st.markdown(f"**Name:** {profile.get('name', 'Unknown')}")
        st.markdown(f"**Bio:** {profile.get('bio', 'No bio available')}")
        st.markdown(f"**Speaking Style:** {profile.get('style', 'Neutral')}")
    
except Exception as e:
    st.error(f"❌ Failed to load the interview persona: {str(e)}")
    st.info("This might be due to missing API keys or network issues.")
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat messages area
st.markdown('<div class="chat-messages">', unsafe_allow_html=True)

if st.session_state.chat_history:
    for i, (role, msg) in enumerate(st.session_state.chat_history):
        if role == "You":
            st.markdown(f'<div class="user-message"><strong>You:</strong> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><strong>{profile.get("name", "Interviewee")}:</strong> {msg}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="bot-message"><em>👋 Hi! I\'m ready to chat. Ask me anything!</em></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Chat input area
st.markdown('<div class="input-container">', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input(
        "Type your message...", 
        key="chat_input",
        placeholder="Ask me about my research, experiences, or anything else..."
    )
with col2:
    send_button = st.button("Send", type="primary", use_container_width=True)

# Handle sending messages
if send_button and user_input:
    with st.spinner("🤔 Thinking..."):
        try:
            response = qa_chain.invoke({"question": user_input})
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Interviewee", response["answer"]))
            st.rerun()
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

# Clear chat button (outside the chat container)
st.markdown('</div>', unsafe_allow_html=True)  # Close the chat container

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
with col2:
    # Download Jamie's transcript
    with open("interview.pdf", "rb") as pdf_file:
        st.download_button(
            label="📄 Download Jamie's Transcript",
            data=pdf_file.read(),
            file_name="Jamie_Interview_Transcript.pdf",
            mime="application/pdf",
            use_container_width=True
        )
with col3:
    st.write("")  # Empty column for spacing
