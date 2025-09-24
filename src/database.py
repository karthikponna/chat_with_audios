from qdrant_client import QdrantClient
from qdrant_client import models

from src.embed import batch_iterate


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