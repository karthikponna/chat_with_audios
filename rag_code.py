from qdrant_client import models
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.llms.ollama import Ollama
import assemblyai as aai
from typing import List, Dict

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

def batch_iterate(lst, batch_size):
    """
    Yield successive n-sized chunks from lst.
    
    Args:
         lst (list): List of items to iterate over.
         batch_size (int): Size of each batch.
    Returns:
         Generator: Yields successive chunks of the list.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    """
    Class for generating text embeddings using HuggingFaceEmbedding.
    
    Args:
         embed_model_name (str): Name of the embedding model.
         batch_size (int): Batch size for embedding.
         chunk_size (int): Maximum number of characters per chunk.
    Returns:
         None
    """
    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", batch_size=32, chunk_size=200):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.chunk_size = chunk_size  # maximum number of characters per chunk
        self.embeddings = []
        
    def _load_embed_model(self):
        """
        Load the HuggingFace embedding model.
        
        Args:
             None
        Returns:
             HuggingFaceEmbedding: Initialized embedding model.
        """
        embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name, 
            trust_remote_code=True, 
            cache_folder='./hf_cache'
        )
        return embed_model

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits a long text into chunks of maximum self.chunk_size characters.
        
        Args:
             text (str): The text to split.
        Returns:
             List[str]: List of text chunks.
        """
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
    def generate_embedding(self, contexts):
        """
        Generate embeddings for a batch of contexts.
        
        Args:
             contexts (list): List of text contexts.
        Returns:
             List: Embeddings for the provided contexts.
        """
        return self.embed_model.get_text_embedding_batch(contexts)
        
    def embed(self, contexts: List[str]):
        """
        Generate embeddings for contexts, chunking them if they exceed the chunk size.
        
        Args:
             contexts (List[str]): List of text contexts.
        Returns:
             None
        """
        # Create a new list to hold all (possibly chunked) contexts.
        all_contexts = []
        for context in contexts:
            if len(context) > self.chunk_size:
                all_contexts.extend(self.chunk_text(context))
            else:
                all_contexts.append(context)
        self.contexts = all_contexts
        
        for batch_context in batch_iterate(self.contexts, self.batch_size):
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)

class QdrantVDB_QB:
    """
    Class to manage Qdrant vector database operations for ingestion and retrieval.
    
    Args:
         collection_name (str): Name of the Qdrant collection.
         vector_dim (int): Dimension of the vector embeddings.
         batch_size (int): Batch size for operations.
    Returns:
         None
    """
    def __init__(self, collection_name, vector_dim=768, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def define_client(self):
        """
        Define and initialize the Qdrant client.
        
        Args:
             None
        Returns:
             None
        """
        self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)

    def clear_collection(self):
        """
        Clear the collection if it exists.
        
        Args:
             None
        Returns:
             None
        """
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

    def create_collection(self):
        """
        Create a new collection in Qdrant if it doesn't exist.
        
        Args:
             None
        Returns:
             None
        """
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.DOT,
                    on_disk=True
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    default_segment_number=5,
                    indexing_threshold=0
                ),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(always_ram=True)
                ),
            )

    def ingest_data(self, embeddata):
        """
        Ingest embedding data into the Qdrant collection.
        
        Args:
             embeddata (EmbedData): Instance containing contexts and their embeddings.
        Returns:
             None
        """
        for batch_context, batch_embeddings in zip(
            batch_iterate(embeddata.contexts, self.batch_size), 
            batch_iterate(embeddata.embeddings, self.batch_size)
        ):
            self.client.upload_collection(
                collection_name=self.collection_name,
                vectors=batch_embeddings,
                payload=[{"context": context} for context in batch_context]
            )
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000)
        )

class Retriever:
    """
    Class to perform semantic search on a Qdrant vector database using embedded data.
    
    Args:
         vector_db (QdrantVDB_QB): Qdrant vector database instance.
         embeddata (EmbedData): Instance for generating text embeddings.
    Returns:
         None
    """
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        """
        Perform a semantic search using a query's embedding.
        
        Args:
             query (str): The search query.
        Returns:
             List: Search results from the vector database.
        """
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)
        result = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            timeout=1000,
        )
        return result
    
class RAG:
    """
    Class to perform Retrieval-Augmented Generation by combining context retrieval with LLM responses.
    
    Args:
         retriever (Retriever): Instance of Retriever for context retrieval.
         llm_name (str): Name of the LLM to use.
    Returns:
         None
    """
    def __init__(self,
                 retriever,
                 llm_name="Meta-Llama-3.1-405B-Instruct"
                 ):
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are Ayush, the author of the LLM Engineer Handbook. "
                "You are friendly, authentic, and direct in your interactions, communicating casually like in a WhatsApp chat. "
                "When a user refers to 'the book', assume they mean the LLM Engineer Handbook. "
                "Provide a concise explanation of the book's purpose, content, and intended audience. "
                "Do not include any pricing, discount, sales information, or coupon codes unless the user explicitly asks about them, "
                "and even then, respond minimally by stating that such details are not available. "
                "If a user asks for a purchase link, always respond with: 'You can check out the book available at Amazon.' "
                "Avoid starting responses with casual greetings like 'Hey!' except in the very first interaction. "
                "Use Chain-of-Thought reasoning to identify the user's intent and filter out unnecessary details before responding. "
                "Do not reveal any internal chain-of-thought or reasoning in your final responses. "
                "Keep your final answer within 50 characters."
            ),
        )
        self.messages = [system_msg, ]
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        # Updated QA prompt to include conversation history
        self.qa_prompt_tmpl_str = (
            "Below is context retrieved from our database with specific details about the book. "
            "Below is the conversation history so far: {conversation_history}\n\n"
            "Use both the context and the conversation history to answer the user's query. "
            "Answer strictly based on the context, conversation history, and your inferences, and do not add any extra details. "
            "Keep the tone friendly, direct, and casual, like chatting on WhatsApp. "
            "Do not include any pricing, discount, sales information, or coupon codes unless the user explicitly asks about them.\n\n"
            "Context information:\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n\n"
            "Conversation History:\n"
            "---------------------\n"
            "{conversation_history}\n"
            "---------------------\n\n"
            "Query: {query}\n"
            "Answer: "
        )

    def _setup_llm(self):
        """
        Set up and return the LLM instance.
        
        Args:
             None
        Returns:
             LLM: Configured LLM instance.
        """
        return SambaNovaCloud(
            model=self.llm_name,
            temperature=0.7,
            context_window=100000,
        )
        # Alternatively, you can use Ollama by uncommenting the below code:
        # return Ollama(model=self.llm_name,
        #               temperature=0.7,
        #               context_window=100000,
        #             )

    def generate_context(self, query):
        """
        Retrieve and combine context from the database for the given query.
        
        Args:
             query (str): The user's query.
        Returns:
             str: Combined context information.
        """
        result = self.retriever.search(query)
        retrieved_docs = [dict(data) for data in result]
        combined_prompt = []
        # Retrieve up to top 4 documents (if available)
        for doc in retrieved_docs[:4]:
            combined_prompt.append(doc["payload"]["context"])
        return "\n\n---\n\n".join(combined_prompt)

    def query(self, query, conversation_history=""):
        """
        Formulate the query using context and conversation history, then return a streaming LLM response.
        
        Args:
             query (str): The user's query.
             conversation_history (str): The conversation history to include.
        Returns:
             Streaming response: The LLM's response stream.
        """
        context = self.generate_context(query=query)
        prompt = self.qa_prompt_tmpl_str.format(
            context=context,
            query=query,
            conversation_history=conversation_history
        )
        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)
        streaming_response = self.llm.stream_complete(user_msg.content)
        return streaming_response

class Transcribe:
    """
    Class to transcribe audio files with speaker labeling using AssemblyAI.
    
    Args:
         api_key (str): AssemblyAI API key.
    Returns:
         None
    """
    def __init__(self, api_key: str):
        """Initialize the Transcribe class with AssemblyAI API key."""
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        """
        Transcribe an audio file and return speaker-labeled transcripts.
        
        Args:
             audio_path (str): Path to the audio file.
        Returns:
             List[Dict[str, str]]: List of dictionaries containing speaker and text information.
        """
        # Configure transcription with speaker labels, expecting 1 speaker for a single-speaker audio
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=1  
        )
        
        # Transcribe the audio
        transcript = self.transcriber.transcribe(audio_path, config=config)
        
        # Extract speaker utterances
        speaker_transcripts = []
        for utterance in transcript.utterances:
            speaker_transcripts.append({
                "speaker": f"Speaker {utterance.speaker}",
                "text": utterance.text
            })
            
        return speaker_transcripts
