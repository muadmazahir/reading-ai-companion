from abc import ABC, abstractmethod
from typing import List


class RAG(ABC):
    """
    Base class for RAG
    """

    @abstractmethod
    def parse_document(self, document_source):
        """
        Parse a document into a structured format to be used for RAG

        :param document_source: The URI to the document to parse
        """
        pass

    @abstractmethod
    def chunk_document(self, document):
        """
        Chunk a document into a list of

        :param document: The document to chunk
        """
        pass

    @abstractmethod
    def embed_and_insert_chunks(self, table_name: str, chunks) -> None:
        """
        Embed a list of chunks

        :param table_name: The name of the table to embed the chunks for. 
            This will be used as the table name in the vector database.
        :param chunks: The chunks to embed
        """
        pass

    @abstractmethod
    def search_table(self, table_name: str, queries: List[str], limit: int) -> List[str]:
        """
        Search the table for the given queries

        :param table_name: The name of the table to search in.
        :param queries: The queries to search for.
        :param limit: The number of results to return.
        """
        pass
