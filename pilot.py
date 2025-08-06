# pilot.py  ▲ run with:  python pilot.py
# ----------------------------------------
# 1-A  Load & clean transcript  (unchanged)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter    import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import json, textwrap
import re

# Load environment variables from .env file
load_dotenv()

doc   = PyPDFLoader("interview.pdf").load()[0]
clean = doc.page_content.replace("researcher:", "").replace("Interviewee:", "")

# ---------- 1-B  ⬅ NEW:  auto-extract persona metadata ----------
# Configure for OpenRouter
openrouter_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_key:
    raise ValueError("Please set OPENROUTER_API_KEY environment variable")

os.environ["OPENAI_API_KEY"] = openrouter_key
extract_llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)   # via OpenRouter
EXTRACT_PROMPT = """
You are a data extractor. Read the interview text delimited by <doc>.
Return strict JSON with keys:
name  – full name or null
bio   – 1-sentence bio (job/age/location if stated) or null
style – 2-3 adjectives describing speaking style or null
<doc>{document}</doc>
"""

meta_json  = extract_llm.invoke(EXTRACT_PROMPT.format(document=textwrap.shorten(clean, 12000))).content

# Extract JSON from markdown code blocks if present
json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', meta_json, re.DOTALL)
if json_match:
    meta_json = json_match.group(1)

try:
    profile    = json.loads(meta_json)
except json.JSONDecodeError as e:
    print(f"JSON parsing failed: {e}")
    print(f"Raw response: {repr(meta_json)}")
    # Fallback to default profile
    profile = {
        "name": "Unknown Speaker",
        "bio": "Interviewee in the provided transcript.",
        "style": "neutral"
    }

# fall-backs for missing fields
profile.setdefault("name",  "Unknown Speaker")
profile.setdefault("bio",   "Interviewee in the provided transcript.")
profile.setdefault("style", "neutral")

# ---------- 1-C  Split into chunks  (same as before) ----------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks   = splitter.create_documents([clean])

# ---------- 2  Embed & index  (unchanged) ----------
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
import os

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Please set PINECONE_API_KEY environment variable")

pc = Pinecone(api_key=pinecone_api_key)

# Try different embedding models in order of preference
embeds = None
dimension = None  # Initialize dimension variable
cohere_api_key = os.getenv("COHERE_API_KEY")

print("🔍 Testing embedding models for best performance...")

# 1. Try Cohere (best performance for semantic search)
if cohere_api_key:
    try:
        embeds = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=cohere_api_key
        )
        # Test the embedding
        test_embedding = embeds.embed_documents(["test"])
        print("✅ Using Cohere embeddings (best performance)")
        dimension = 1024  # Cohere embed-english-v3.0 dimension
    except Exception as e:
        print(f"❌ Cohere failed: {e}")

# 2. Try reliable local sentence-transformers model first
if embeds is None:
    print("🔄 Using local sentence-transformers embeddings...")
    try:
        # Use the most reliable sentence-transformers model
        embeds = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        # Test the embedding
        test_embedding = embeds.embed_documents(["test"])
        print("✅ Using all-MiniLM-L6-v2 (reliable local model)")
        dimension = 384
    except Exception as e:
        print(f"❌ all-MiniLM-L6-v2 failed: {e}")
        try:
            # Try the better performing model
            embeds = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Test the embedding
            test_embedding = embeds.embed_documents(["test"])
            print("✅ Using BAAI/bge-small-en-v1.5 (best local model)")
            dimension = 384
        except Exception as e2:
            print(f"❌ BAAI/bge-small-en-v1.5 also failed: {e2}")

# 3. Try OpenRouter embeddings as last resort
if embeds is None:
    try:
        embeds = OpenAIEmbeddings(
            base_url="https://openrouter.ai/api/v1",
            model="text-embedding-3-small"
        )
        # Test the embedding
        test_embedding = embeds.embed_documents(["test"])
        print("✅ Using OpenRouter embeddings")
        dimension = 1536  # text-embedding-3-small dimension
    except Exception as e:
        print(f"❌ OpenRouter embeddings failed: {e}")

# Ensure we have an embedding model and dimension
if embeds is None or dimension is None:
    raise ValueError("No embedding model could be initialized. Please check your internet connection and try again.")

print(f"📊 Using embedding model with dimension: {dimension}")

# Create or get Pinecone index
index_name = "persona-pilot"
try:
    # Check if index exists and has the right dimension
    existing_indexes = pc.list_indexes()
    index_exists = index_name in [idx.name for idx in existing_indexes]
    
    if index_exists:
        # Check if the existing index has the right dimension
        index_description = pc.describe_index(index_name)
        existing_dimension = index_description.dimension
        
        if existing_dimension != dimension:
            print(f"⚠️  Existing index has dimension {existing_dimension}, but we need {dimension}")
            print(f"🔄 Deleting existing index '{index_name}' and creating new one...")
            pc.delete_index(index_name)
            index_exists = False
        else:
            print(f"Using existing Pinecone index: {index_name}")
    
    if not index_exists:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=CloudProvider.AWS,
                region=AwsRegion.US_EAST_1
            )
        )
        print(f"✅ Created Pinecone index with dimension: {dimension}")
    
    # Get the index
    index = pc.Index(index_name)
    
    # Create vector store
    vdb = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeds,
        index_name=index_name
    )
    print("✅ Pinecone vector store created successfully")
    
except Exception as e:
    print(f"❌ Pinecone setup failed: {e}")
    raise

retriever = vdb.as_retriever(search_kwargs={"k": 3})

# ---------- 3  Build dynamic system prompt  (changed) ----------
from langchain.prompts import PromptTemplate

# Create a system message with the profile information
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

# ---------- 4  Ask questions using the persona-based chain ----------
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

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

print("✅ Persona Q&A ready.")

# Only run the interactive loop if this script is run directly
if __name__ == "__main__":
    print("Ask anything to the interviewee. Type 'exit' to quit.\n")
    
    while True:
        user_question = input("You: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        
        response = qa_chain.invoke({"question": user_question})
        print(f"\n{profile['name']}: {response['answer']}\n")
else:
    # When imported by Streamlit, just print a success message
    print("✅ Persona Q&A ready for Streamlit interface.")
