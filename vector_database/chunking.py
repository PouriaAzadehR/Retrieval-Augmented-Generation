from typing import List, Optional
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


EMBEDDING_MODEL_NAME = "thenlper/gte-small"


class ChunkingDocument:
    separators = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(self, chunk_size, knowledge_base, tokenizer_name):
        self.chunk_size = chunk_size
        self.knowledge_base = knowledge_base
        self.tokenizer_name = tokenizer_name
        self.chunks = self.knowledge_base_chunking_to_document()

    def knowledge_base_chunking_to_document(self) -> List[LangchainDocument]:
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.tokenizer_name),
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=ChunkingDocument.separators,
        )

        docs_processed = []
        for doc in self.knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)
        return docs_processed_unique
