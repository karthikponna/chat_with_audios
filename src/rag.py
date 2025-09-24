import os
from qdrant_client import models
from llama_index.llms.sambanovasystems import SambaNovaCloud
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")

SYSTEM_PROMPT = (
    """
You are an intelligent audio conversation assistant. You analyze transcribed audio content and provide helpful, accurate responses based on the conversation context. You maintain a friendly and conversational tone while being precise and informative.
"""
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
    def __init__(self, retriever, llm_name):

        # System prompt 
        self.system_prompt = ChatMessage(
            role=MessageRole.SYSTEM,
            content=SYSTEM_PROMPT,
        )

        self.llm_name = llm_name
        self.llm = self._setup_llm()
        self.retriever = retriever
        
        # User prompt
        self.user_prompt = (
        "Here is the data for generating your response:\n"
        "Audio Transcript Context: {context}\n"
        "Conversation History: {conversation_history}\n"
        "User Query: {query}\n"
        "NOTE: Use the audio transcript context and conversation history to provide an accurate and helpful response. Keep your tone friendly and conversational while being informative and precise."
        )
        
    
    def _setup_llm(self):
        """
        Set up and return the LLM instance.
        """
        return SambaNovaCloud(
            api_key = sambanova_api_key,
            model=self.llm_name,
            temperature=0.0,
            context_window=100000,
        )


    def generate_context(self, query):
        """
        Retrieve and combine context from the database using the (refined) query.
        """
        result = self.retriever.search(query)
        retrieved_docs = [dict(data) for data in result]
        combined_search_results = []
        # Retrieve up to top 4 documents 
        for doc in retrieved_docs[:4]:
            combined_search_results.append(doc["payload"]["context"])

        print(len(combined_search_results))
        return "\n\n---\n\n".join(combined_search_results)


    def query(self, query, conversation_history=""):
        """
        First refine the query, then use the refined version to retrieve context and generate an answer.
        """
        
        context = self.generate_context(query=query)
        prompt = self.user_prompt.format(
            context=context,
            query=query,
            conversation_history=conversation_history
        )
    
        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)

        messages = [self.system_prompt, user_msg]
        streaming_response = self.llm.stream_chat(messages)
        return streaming_response


