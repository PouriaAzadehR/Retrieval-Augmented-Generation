from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

from vector_database.chunking import ChunkingDocument
from vector_database.raw_knowledge_base import RAW_KNOWLEDGE_BASE

EMBEDDING_MODEL_NAME = "thenlper/gte-small"


class VectorDatabase:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
        )
        self.chunkingDocument = ChunkingDocument(512, RAW_KNOWLEDGE_BASE, EMBEDDING_MODEL_NAME).chunks
        self.knowledgeVectorBase = FAISS.from_documents(
            self.chunkingDocument, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
        )
