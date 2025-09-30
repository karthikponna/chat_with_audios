# RAG Meets Audio: Chat with Your Recordings via AssemblyAI and Qwen3-32B

<video src="https://github.com/user-attachments/assets/3590e0cf-9576-4abc-97a7-e2b63274d9c0"/></video>

[Check out the Blog](https://www.analyticsvidhya.com/blog/2025/03/audio-rag/)

This project combines the power of Retrieval-Augmented Generation (RAG) with AssemblyAI's transcription capabilities, enabling you to interact with audio recordings as if they were conversational text. By leveraging Qwen3-32b for natural language understanding, this solution efficiently retrieves and answers queries based on your audio content.

# 🚀 Features
- **Audio Transcription** using AssemblyAI for accurate speech-to-text conversion.
- **Qdrant Vector Database** for efficient retrieval and semantic search.
- **DeepSeek R1** via **SambaNova Cloud** for powerful language model responses.
- Seamless integration of transcription and **RAG (Retrieval-Augmented Generation)** for improved context-aware conversations.

# 🧠 How It Works
1. **Transcription:** AssemblyAI transcribes your audio file, extracting speaker information for better clarity.
2. **Embedding Generation:** Text data is embedded using HuggingFace's `BAAI/bge-large-en-v1.5` model.
3. **Vector Search:** Qdrant's vector database efficiently retrieves relevant context from indexed data.
4. **RAG Model Response:** DeepSeek R1 generates accurate and context-aware responses based on retrieved content.

# **⚙️ Installation**

## 1. **Install uv**:
```bash
# For macOS:
brew update
brew install uv

# For Windows - Open PowerShell as Administrator and run the below command:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[More on installing uv](https://docs.astral.sh/uv/getting-started/installation/)


# **🎯 Getting Started**

## 1. Clone the repository
   ```bash
   git clone https://github.com/karthikponna/chat_with_audios.git
   cd chat_with_audios
   ```
## 2. Install dependencies

First deactivate any active virtual environment

```bash
deactivate
```

To install the dependencies and activate the virtual environment, run the following command

```bash
uv venv .venv
uv sync
```

## 3. Environment Configuration

Before running any command, you have to set up your environment:
1. Create your environment file:
   ```bash
   cp .env.example .env
   ```

Add AssemblyAI and Sambanova API keys in the .env file.

# 🏗️ Set Up Your Local Infrastructure

We use Docker to set up the local infrastructure (Qdrant, Streamlit).

> [!WARNING]
> Before running the command below, ensure you do not have any processes running on port `6333` and `6334` (Qdrant) and `8501` (Streamlit).

To start the Docker infrastructure, run:
```bash
docker compose build
```

To stop the Docker infrastructure, run:
```bash
docker compose stop
```

# ⚡️ Running the Code

Run the below command to start the app:
```bash
docker compose up
```

# 🙌 Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

If you find any bugs and know how to fix them. You can always contribute by:

- Forking the repository
- Fixing the bug
- Creating a pull request
