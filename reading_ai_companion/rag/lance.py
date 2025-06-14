import logging
import os
import re
from pathlib import Path
from typing import List

import lancedb
import tiktoken
from docling.chunking import HybridChunker
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc.document import DoclingDocument
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import LanceTable

from reading_ai_companion.rag.base import RAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the OpenAI embedding function
EMBEDDING_FUNC = get_registry().get('openai').create(name='text-embedding-3-large')


# Define a simplified metadata schema
class ChunkMetadata(LanceModel):
    """
    You must order the fields in alphabetical order.
    This is a requirement of the Pydantic implementation.
    """

    filename: str | None
    page_numbers: List[int] | None
    title: str | None


class Chunks(LanceModel):
    text: str = EMBEDDING_FUNC.SourceField()
    vector: Vector(EMBEDDING_FUNC.ndims()) = EMBEDDING_FUNC.VectorField()   # type: ignore
    metadata: ChunkMetadata


class LanceRAG(RAG):
    """
    Class to handle RAGs
    """

    def __init__(self) -> None:
        assert os.environ.get('LANCE_DB_URI') is not None, 'LANCE_DB_URI is not set'
        self.lance_db = lancedb.connect(os.environ.get('LANCE_DB_URI'))

    def parse_document(self, document_source: Path | str | DocumentStream) -> DoclingDocument:
        """
        Parse a document into a structured format to be used for RAG

        :param document_source: The URI to the document to parse
        """
        converter = DocumentConverter()
        return converter.convert(document_source).document

    def chunk_document(self, document: DoclingDocument) -> List[DocChunk]:
        """
        Chunk a document into a list of

        :param document: The document to chunk as a DoclingDocument object
        """
        tokenizer = OpenAITokenizer(
            tokenizer=tiktoken.encoding_for_model('text-embedding-3-small'),
            max_tokens=8191,
        )
        chunker = HybridChunker(tokenizer=tokenizer)
        chunk_list = list(chunker.chunk(document))
        return chunk_list

    def embed_and_insert_chunks(self, table_name: str, chunks: List[DocChunk]) -> None:
        """
        Embed a list of chunks

        :param chunks: The chunks to embed
        """
        table_name = self._sanitize_table_name_for_lancedb(table_name)
        table = self._verify_table(table_name)
        if table is None:
            table = self.lance_db.create_table(table_name, schema=Chunks)
            logger.info('Table created: %s', table_name)

        processed_chunks = [
            {
                'text': chunk.text,
                'metadata': {
                    'filename': chunk.meta.origin.filename,
                    'page_numbers': [
                        page_no
                        for page_no in sorted(set(prov.page_no for item in chunk.meta.doc_items for prov in item.prov))
                    ]
                    or None,
                    'title': chunk.meta.headings[0] if chunk.meta.headings else None,
                },
            }
            for chunk in chunks
        ]

        table.add(processed_chunks)

    def search_table(self, table_name: str, queries: List[str], limit: int = 3) -> List[str]:
        """
        Do a vector search on a table using a query and return the texts that are the closest matches

        :param table_name: The name of the table to search in
        :param queries: The queries to search for
        """

        table_name = self._sanitize_table_name_for_lancedb(table_name)

        table = self._verify_table(table_name)
        if table is None:
            raise ValueError(
                f'Table {table_name} does not exist. Knowledge base not setup for this book. '
                'Please run Companion.setup_knowledge_base() to setup the knowledge base.'
            )

        unique_texts = []

        for query in queries:
            item_list = table.search(query=query, query_type='vector').limit(limit).to_list()
            for item in item_list:
                if item['text'] not in unique_texts:
                    unique_texts.append(item['text'])
                    logger.info('Text retrieved from knowledge base: %s', item['metadata'])
        return unique_texts

    def _verify_table(self, table_name: str) -> LanceTable | None:
        """
        Verify the table exists and return it if it does.
        """
        table_list = self.lance_db.table_names()
        if table_name in table_list:
            table = self.lance_db.open_table(table_name)
            logger.info('Table opened: %s', table_name)
            return table
        return None

    @staticmethod
    def _sanitize_table_name_for_lancedb(table_name: str) -> str:
        """
        Converts a provided table name to a valid LanceDB table name.

        - Keeps only alphanumeric characters, underscores, hyphens, and periods.
        - Replaces spaces with underscores.
        - Removes subtitles (text after a colon, dash, or em-dash).
        - Converts to lowercase.

        Example:
            'The Wealth of Nations' -> 'the_wealth_of_nations'
            'Sapiens: A Brief History of Humankind' -> 'sapiens'
        """
        # Remove subtitle (text after colon, dash, or em dash)
        name = re.split(r'[:\-—]', table_name)[0]

        # Strip leading/trailing whitespace
        name = name.strip()

        # Replace spaces with underscores
        name = name.replace(' ', '_')

        # Remove any characters that are not allowed
        name = re.sub(r'[^a-zA-Z0-9_.-]', '', name)

        # Convert to lowercase
        return name.lower()
