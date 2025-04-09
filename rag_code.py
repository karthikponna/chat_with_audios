from qdrant_client import models
from qdrant_client import QdrantClient
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.sambanovasystems import SambaNovaCloud

from file_handling import load_global_context

import assemblyai as aai
from typing import List, Dict
import os

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)


        
def batch_iterate(lst, batch_size):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]

class EmbedData:
    """
    Class for generating text embeddings using HuggingFaceEmbedding.
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
        """
        embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name, 
            trust_remote_code=True, 
            cache_folder='./hf_cache'
        )
        return embed_model

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits a long text into chunks.
        """
        return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        
    def generate_embedding(self, contexts):
        """
        Generate embeddings for a batch of contexts.
        """
        return self.embed_model.get_text_embedding_batch(contexts)
        
    def embed(self, contexts: List[str]):
        """
        Process and embed contexts.
        """
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
    Class to manage Qdrant vector database operations.
    """
    def __init__(self, collection_name, vector_dim=768, batch_size=512):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim

    def define_client(self):
        """
        Define and initialize the Qdrant client.
        """
        self.client = QdrantClient(url="http://localhost:6333", prefer_grpc=True)

    def clear_collection(self):
        """
        Clear the collection if it exists.
        """
        if self.client.collection_exists(collection_name=self.collection_name):
            self.client.delete_collection(collection_name=self.collection_name)

    def create_collection(self):
        """
        Create a new collection in Qdrant if it doesn't exist.
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
    Class to perform semantic search on a Qdrant vector database.
    """
    def __init__(self, vector_db, embeddata):
        self.vector_db = vector_db
        self.embeddata = embeddata

    def search(self, query):
        """
        Perform a semantic search using a query's embedding.
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
    Class to perform Retrieval-Augmented Generation.
    """
    def __init__(self,
                 retriever,
                 llm_name="Meta-Llama-3.1-405B-Instruct",
                 ):
        # System prompt for final answer generation
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "You are Ayush, the author of the LLM Engineer Handbook. "
                "You are friendly, authentic, and direct in your interactions. " 
                "Provide a concise explanation of the book's purpose and content when needed. "
                "Do not include any pricing, discount, or sales information unless explicitly asked. "
                "Avoid casual greetings after the first interaction. "
                "Keep your final answer within 50 characters."
            ),
        )

        self.messages = system_msg
        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        
        # Updated QA prompt to include conversation history
        self.qa_prompt_tmpl_str = (
            "Below is context retrieved from our database with specific details about the book. "
            "Context information:\n"
            "---------------------\n"
            "{context}\n"
            "This is the conversation history so far: {conversation_history}\n\n"
            "Use both the context and the conversation history to answer the user's query. "
            "---------------------\n\n"
            "User Query: {query}\n" 
            "Answer strictly based on the context, conversation history, and your inferences. "
            "Keep the tone friendly, direct, and casual. "
            "Do not include any pricing, discount, or sales information unless explicitly asked.\n\n" 
            "Answer: "
        )
        self.global_context = load_global_context("context_refining_query/global_context_file.txt")
        print(self.global_context)
        
    
    def _setup_llm(self):
        """
        Set up and return the LLM instance.
        """
        return SambaNovaCloud(
            model=self.llm_name,
            temperature=0.4,
            context_window=100000,
        )
        # Alternatively, to use Ollama, comment above and use below:
        # return Ollama(model=self.llm_name,
        #               temperature=0.7,
        #               context_window=100000,
        #             )
    
    def refine_query(self, query: str) -> str:
        
        refine_prompt = (
            "You are an assistant tasked with refining user queries relative to the context provided. "
            "Below is the global context:\n"
            "---------------------\n"
            f"{self.global_context}\n"
            "---------------------\n\n"
            "User Query: " + query + "\n\n"
            "If the user query is not at all relevant to the context, output should be something like: \n"
            "\"Interesting question, but that's a bit outside my knowledge. Want to know more about LLM Engineer Handbook instead?\"\n\n"
            "Otherwise, if it is relevant, output a clear and specific query relevant to the context.\n"
            "DO NOT include any internal chain-of-thought or explanation. Only output the final result.\n\n"
            "Final Refined Query:"
        )
        
        refined_query_resp = self.llm.complete(refine_prompt)

        if hasattr(refined_query_resp, "text"):
            refined_query_str = refined_query_resp.text
        else:
            refined_query_str = str(refined_query_resp)
        
        
        return refined_query_str.strip()


    def generate_context(self, query):
        """
        Retrieve and combine context from the database using the (refined) query.
        """
        result = self.retriever.search(query)
        retrieved_docs = [dict(data) for data in result]
        combined_prompt = []
        # Retrieve up to top 4 documents 
        for doc in retrieved_docs[:4]:
            combined_prompt.append(doc["payload"]["context"])
        return "\n\n---\n\n".join(combined_prompt)

    def query(self, query, conversation_history=""):
        """
        First refine the query, then use the refined version to retrieve context and generate an answer.
        """
        refined_query = self.refine_query(query)
        print(f"Refined Query: {refined_query}")
        
        context = self.generate_context(query=refined_query)
        prompt = self.qa_prompt_tmpl_str.format(
            context=context,
            query=refined_query,
            conversation_history=conversation_history
        )
        print(f"context: {context}")
        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)
        streaming_response = self.llm.stream_complete(user_msg.content)
        return streaming_response

class Transcribe:
    """
    Class to transcribe audio files with speaker labeling using AssemblyAI.
    """
    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        self.transcriber = aai.Transcriber()
        
    def transcribe_audio(self, audio_path: str) -> List[Dict[str, str]]:
        """
        Transcribe an audio file and return speaker-labeled transcripts.
        """
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=1  
        )
        transcript = self.transcriber.transcribe(audio_path, config=config)
        speaker_transcripts = []
        for utterance in transcript.utterances:
            speaker_transcripts.append({
                "speaker": f"Speaker {utterance.speaker}",
                "text": utterance.text
            })
        return speaker_transcripts
