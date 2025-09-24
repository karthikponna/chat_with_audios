from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Dict


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