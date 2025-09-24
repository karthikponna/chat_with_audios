# RAG Meets Audio: Chat with Your Recordings via AssemblyAI and DeepSeek R1

[Watch the Demo Video](https://youtu.be/Qmh2p0XFqiQ)

[Check out the Blog](https://www.analyticsvidhya.com/blog/2025/03/audio-rag/)

This project combines the power of Retrieval-Augmented Generation (RAG) with AssemblyAI's transcription capabilities, enabling you to interact with audio recordings as if they were conversational text. By leveraging DeepSeek R1 (or Meta-Llama 3.1-405B-Instruct) for natural language understanding, this solution efficiently retrieves and answers queries based on your audio content.

## 🚀 Features
- **Audio Transcription** using AssemblyAI for accurate speech-to-text conversion.
- **Qdrant Vector Database** for efficient retrieval and semantic search.
- **DeepSeek R1** via **SambaNova Cloud** for powerful language model responses.
- Seamless integration of transcription and **RAG (Retrieval-Augmented Generation)** for improved context-aware conversations.

## 🧠 How It Works
1. **Transcription:** AssemblyAI transcribes your audio file, extracting speaker information for better clarity.
2. **Embedding Generation:** Text data is embedded using HuggingFace's `BAAI/bge-large-en-v1.5` model.
3. **Vector Search:** Qdrant's vector database efficiently retrieves relevant context from indexed data.
4. **RAG Model Response:** DeepSeek R1 generates accurate and context-aware responses based on retrieved content.

## 📂 Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/karthikponna/chat_with_audios.git
   cd chat_with_audios
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your `.env` file with the required API keys:
   ```env
   ASSEMBLYAI_API_KEY="your_api_key_here"
   SAMBANOVA_API_KEY="your_api_key_here"
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## 🙌 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.


#### NOTES:

docker ps 
dokcer stop <containerid>
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

Open terminal 2 and run streamlit app.py
uv run streamlit run app.py
