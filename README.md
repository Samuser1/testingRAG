# 🎙️ Persona Interview Simulator

A Streamlit app that creates an AI persona based on interview transcripts, allowing you to have conversations with the interviewee.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the project root with:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   COHERE_API_KEY=your_cohere_api_key_here
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Initialize the system:**
   - The app will show a simple "Initialize AI System" button
   - Click it to start the RAG setup process
   - Watch the progress as it loads the transcript, creates embeddings, and sets up the database
   - Once complete, you can start chatting with the AI persona

## 🔑 Getting API Keys

- **OpenRouter API Key:** Sign up at [openrouter.ai](https://openrouter.ai) to get access to GPT-4 and other models
- **Pinecone API Key:** Sign up at [pinecone.io](https://pinecone.io) for vector database storage
- **Cohere API Key:** Sign up at [cohere.ai](https://cohere.ai) for better embeddings (optional)

## 📁 Files

- `app.py` - Main Streamlit application
- `pilot.py` - Backend logic for persona creation and Q&A
- `interview.pdf` - Interview transcript to create the persona from
- `requirements.txt` - Python dependencies

## 🎯 Features

- Creates an AI persona from interview transcripts
- Interactive chat interface with modern UI
- **On-demand RAG system initialization** - no more slow loading!
- **Real-time progress tracking** during system setup
- **Cohere AI embeddings** for better semantic search
- Persistent conversation history
- Error handling and user feedback
- Responsive design

## 🐛 Troubleshooting

If you see a blank page:
1. Check that all required API keys are set in your `.env` file
2. Ensure you have an active internet connection
3. Verify that `interview.pdf` exists in the project directory
4. Check the terminal for any error messages 