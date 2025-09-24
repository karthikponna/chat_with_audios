from qdrant_client import models
from llama_index.llms.sambanovasystems import SambaNovaCloud

from file_handling import load_global_context

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)

import os

sambanova_api_key = os.getenv("SAMBANOVA_API_KEY")


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
            api_key = sambanova_api_key,
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


