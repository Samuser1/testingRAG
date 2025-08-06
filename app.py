import streamlit as st
import os
from dotenv import load_dotenv
import time

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
    .progress-step {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .progress-step.completed {
        background: #d4edda;
        border-color: #c3e6cb;
    }
    .progress-step.active {
        background: #fff3cd;
        border-color: #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

# Check if required environment variables are set
required_vars = ["OPENROUTER_API_KEY", "PINECONE_API_KEY", "COHERE_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    st.info("Please create a `.env` file with the following variables:")
    st.code(f"""
OPENROUTER_API_KEY=your_openrouter_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
    """)
    st.stop()

# Initialize session state
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "setup_complete" not in st.session_state:
    st.session_state.setup_complete = False

def initialize_rag_system():
    """Initialize the RAG system with simple progress tracking"""
    progress_placeholder = st.empty()
    
    # Step 1: Loading and cleaning transcript
    with progress_placeholder.container():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <h4>Loading interview transcript...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import ChatOpenAI
    import json
    import textwrap
    import re
    
    doc = PyPDFLoader("interview.pdf").load()[0]
    clean = doc.page_content.replace("researcher:", "").replace("Interviewee:", "")
    
    time.sleep(0.5)  # Small delay for visual feedback
    
    # Step 2: Extracting persona metadata
    with progress_placeholder.container():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <h4>Analyzing interview content...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # Configure for OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    os.environ["OPENAI_API_KEY"] = openrouter_key
    extract_llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1"
    )
    
    EXTRACT_PROMPT = """
    You are a data extractor. Read the interview text delimited by <doc>.
    Return strict JSON with keys:
    name  – full name or null
    bio   – 1-sentence bio (job/age/location if stated) or null
    style – 2-3 adjectives describing speaking style or null
    <doc>{document}</doc>
    """
    
    meta_json = extract_llm.invoke(EXTRACT_PROMPT.format(document=textwrap.shorten(clean, 12000))).content
    
    # Extract JSON from markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', meta_json, re.DOTALL)
    if json_match:
        meta_json = json_match.group(1)
    
    try:
        profile = json.loads(meta_json)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        profile = {
            "name": "Jamie",
            "bio": "Interviewee in the provided transcript.",
            "style": "neutral"
        }
    
    # Fall-backs for missing fields
    if not profile.get("name") or profile.get("name") == "null" or profile.get("name") is None:
        profile["name"] = "Jamie"  # Hardcode Jamie as fallback
    profile.setdefault("bio", "Interviewee in the provided transcript.")
    profile.setdefault("style", "neutral")
    
    time.sleep(0.5)
    
    # Step 3: Creating text chunks
    with progress_placeholder.container():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <h4>Preparing searchable content...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.create_documents([clean])
    
    time.sleep(0.5)
    
    # Step 4: Creating embeddings with Cohere
    with progress_placeholder.container():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <h4>Creating AI embeddings...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    from langchain_cohere import CohereEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
    
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY is required")
    
    embeds = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key
    )
    dimension = 1024  # Cohere embed-english-v3.0 dimension
    
    time.sleep(0.5)
    
    # Step 5: Setting up Pinecone database
    with progress_placeholder.container():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <h4>Setting up database...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Create or get Pinecone index
    index_name = "persona-pilot"
    try:
        existing_indexes = pc.list_indexes()
        index_exists = index_name in [idx.name for idx in existing_indexes]
        
        if index_exists:
            index_description = pc.describe_index(index_name)
            existing_dimension = index_description.dimension
            
            if existing_dimension != dimension:
                pc.delete_index(index_name)
                index_exists = False
        
        if not index_exists:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_EAST_1
                )
            )
        
        # Create vector store
        vdb = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeds,
            index_name=index_name
        )
        
    except Exception as e:
        st.error(f"❌ Pinecone setup failed: {e}")
        return False
    
    time.sleep(0.5)
    
    # Step 6: Building the Q&A system
    with progress_placeholder.container():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">⏳</div>
            <h4>Building conversation system...</h4>
        </div>
        """, unsafe_allow_html=True)
    
    from langchain.prompts import PromptTemplate
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    
    retriever = vdb.as_retriever(search_kwargs={"k": 3})
    
    # Create system message with profile information
    system_message = f"""
    You are {profile['name']}, speaking in first person.
    Base every answer on the excerpts provided below. If a topic is missing,
    answer with your best reasoning but stay consistent with your past statements;
    If unsure, say so. If not found, answer using your best reasoning as {profile['name']}. 
    Persona bio: {profile['bio']}. Style: {profile['style']}.
    Base your answer on the transcript excerpts when relevant.
    If the answer is not covered, respond in character as {profile['name']}, using your lived experience and personality.
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=system_message + """
    
    Relevant excerpts from the interview:
    {context}
    
    Chat History:
    {chat_history}
    
    Human: {question}
    Assistant:"""
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(
            model_name="gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1"
        ),
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    time.sleep(0.5)
    
    # Final success message
    with progress_placeholder.container():
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; margin-bottom: 10px;">✅</div>
            <h4>Ready to chat!</h4>
            <p>You can now have a conversation with {name}</p>
        </div>
        """.format(name=profile['name']), unsafe_allow_html=True)
    
    # Store in session state
    st.session_state.qa_chain = qa_chain
    st.session_state.profile = profile
    st.session_state.setup_complete = True
    
    time.sleep(1)
    st.rerun()

# Main app logic
if not st.session_state.setup_complete:
    # Show initial setup screen
    st.title("🎙️ Persona Interview Simulator")
    st.markdown("### Ready to chat with your interview persona?")
    st.markdown("Click the button below to initialize the AI system. This will:")
    st.markdown("- 📄 Load and process the interview transcript")
    st.markdown("- 👤 Extract persona information")
    st.markdown("- 🧠 Create embeddings using Cohere AI")
    st.markdown("- 🗄️ Set up the vector database")
    st.markdown("- 🤖 Build the conversational AI system")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Initialize AI System", type="primary", use_container_width=True):
            initialize_rag_system()
    
    st.markdown("---")
    st.markdown("**Note:** This process may take 30-60 seconds on first run as it sets up the entire RAG system.")
    
else:
    # Show chat interface
    profile = st.session_state.profile
    qa_chain = st.session_state.qa_chain
    

    
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
        if st.button("🔄 Reset System", use_container_width=True):
            st.session_state.qa_chain = None
            st.session_state.profile = None
            st.session_state.chat_history = []
            st.session_state.setup_complete = False
            st.rerun()
