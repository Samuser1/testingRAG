# pilot.py - Utility functions for the Persona Interview Simulator
# ----------------------------------------
# This module now contains utility functions that can be imported by app.py
# The main execution logic has been moved to app.py for better user experience

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion
import os
import json
import textwrap
import re

def load_and_clean_transcript(pdf_path="interview.pdf"):
    """Load and clean the interview transcript"""
    doc = PyPDFLoader(pdf_path).load()[0]
    clean = doc.page_content.replace("researcher:", "").replace("Interviewee:", "")
    return clean

def extract_persona_metadata(clean_text, openrouter_key):
    """Extract persona information from the interview text"""
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
    
    meta_json = extract_llm.invoke(EXTRACT_PROMPT.format(document=textwrap.shorten(clean_text, 12000))).content
    
    # Extract JSON from markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', meta_json, re.DOTALL)
    if json_match:
        meta_json = json_match.group(1)
    
    try:
        profile = json.loads(meta_json)
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        profile = {
            "name": "Unknown Speaker",
            "bio": "Interviewee in the provided transcript.",
            "style": "neutral"
        }
    
    # Fall-backs for missing fields
    profile.setdefault("name", "Unknown Speaker")
    profile.setdefault("bio", "Interviewee in the provided transcript.")
    profile.setdefault("style", "neutral")
    
    return profile

def create_text_chunks(clean_text, chunk_size=800, chunk_overlap=100):
    """Split text into chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.create_documents([clean_text])
    return chunks

def setup_cohere_embeddings(cohere_api_key):
    """Set up Cohere embeddings"""
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY is required")
    
    embeds = CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=cohere_api_key
    )
    return embeds, 1024  # Cohere embed-english-v3.0 dimension

def setup_pinecone_vectorstore(chunks, embeds, dimension, pinecone_api_key, index_name="persona-pilot"):
    """Set up Pinecone vector store"""
    pc = Pinecone(api_key=pinecone_api_key)
    
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
        
        return vdb
        
    except Exception as e:
        raise Exception(f"Pinecone setup failed: {e}")

def create_qa_chain(vdb, profile, openrouter_key):
    """Create the conversational Q&A chain"""
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
    
    return qa_chain

# Legacy support - if this file is run directly, show a message
if __name__ == "__main__":
    print("✅ Persona Q&A utilities ready for import.")
    print("This module is now used by app.py for the Streamlit interface.")
    print("Run 'streamlit run app.py' to start the application.")
