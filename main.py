from ragatouille import RAGPretrainedModel

from QA.QAwithRAG import answer_with_rag
from models.Reader_LLM import ReaderLLM
from vector_database.vector_db import VectorDatabase

question = "how to create a pipeline object?"
vectorDatabase = VectorDatabase().knowledgeVectorBase
readerLLM = ReaderLLM().llm_pipeline
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

answer, relevant_docs = answer_with_rag(question, readerLLM, vectorDatabase, reranker=RERANKER)
