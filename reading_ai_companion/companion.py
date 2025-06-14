import logging
import os
from typing import Any, Dict, List

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool

from reading_ai_companion.knowledge_collector.core import CoreKnowledgeCollector
from reading_ai_companion.llm_schema import (
    COMPANION_SYSTEM_PROMPT,
    CONCEPT_EXTRACTOR_SYSTEM_PROMPT,
    QUERY_CONSTRUCTION_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    BookExists,
    ConceptExtractor,
    VectorDatabaseQuery,
)
from reading_ai_companion.rag import LanceRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Companion:
    """
    A AI companion to help you understand books.
    """
    def __init__(self, book_name: str, author: str | None = None, model: str = 'o4-mini'):
        assert os.environ.get('OPENAI_API_KEY') is not None, 'OPENAI_API_KEY is not set'

        self.model = model

        verifier_agent = Agent(
            name='Verifier', instructions=VERIFIER_SYSTEM_PROMPT, model=model, output_type=BookExists
        )
        input_dict = {'Book': book_name, 'Author': author}
        verifier_result = Runner.run_sync(verifier_agent, str(input_dict))
        if not verifier_result.final_output.exists:
            raise ValueError(f'The book {book_name} is not a real, published book.')

        self.book_name = verifier_result.final_output.book_name
        self.author = verifier_result.final_output.author

        logger.info('Book name: %s; Author: %s', self.book_name, self.author)

        formatted_prompt = COMPANION_SYSTEM_PROMPT.format(book_name=book_name, author=self.author)
        self.agent = Agent(name='Assistant', instructions=formatted_prompt, model=model)
        self.agent_context: List[Dict[str, Any]] = []

    def explain(self, use_knowledge_base: bool = False, num_concepts: int = 3, chapter: str | None = None):
        """
        Explain the book based.

        :param use_knowledge_base: Whether to use the knowledge base to answer the question.
        :param num_concepts: The number of concepts to use to construct the queries to search the knowledge base.
        :param chapter: The name of the chapter to explain. If None, explain the entire book.
        """
        relevant_concepts = self._get_relevant_concepts(chapter=chapter, num_concepts=num_concepts)
        logger.info('Concepts retrieved: %s', relevant_concepts)

        if chapter is None:
            agent_query = (
                f'Explain the book {self.book_name} by {self.author}. '
                f'Focus on the following concepts: {relevant_concepts}'
            )
        else:
            agent_query = (
                f'Explain the chapter {chapter} of the book {self.book_name} by {self.author}. '
                f'Focus on the following concepts: {relevant_concepts}'
            )

        if use_knowledge_base:
            queries = self._construct_queries_to_search_knowledge_base(relevant_concepts)
            logger.info('Queries to search the knowledge base: %s', queries)

            self.agent.tools = [Companion.get_relevant_info_from_knowledge_base]

            agent_query += (
                ' Get the relevant information from the knowledge base using the '
                'Companion.get_relevant_info_from_knowledge_base tool and use it in the explanation.'
            )
            run_config = RunConfig(model_settings=ModelSettings(tool_choice='required'))
        else:
            run_config = None

        result = Runner.run_sync(self.agent, agent_query, run_config=run_config)

        # set tools to empty list to avoid using tools in future queries
        self.agent.tools = []

        # update agent context so that any follow up query has relevant context
        self.agent_context = result.to_input_list()

        return result.final_output

    def query(self, query: str):
        """
        Ask a custom query about the book.

        :param query: The query to ask about the book.
        """
        prompt = self.agent_context + [{'role': 'user', 'content': query}]
        result = Runner.run_sync(self.agent, input=prompt)

        # update agent context so that any follow up query has relevant context
        self.agent_context = result.to_input_list()

        return result.final_output

    def setup_knowledge_base(self, num_concepts: int = 10, num_items_per_concept: int = 3):
        """
        Setup the knowledge base by retrieving the relevant literature for the book and storing in a vector database.

        It first derives the most important concepts for the book 
        and then retrieves the relevant literature for each concept.

        The literatures is journal articles, books, essays, etc.

        :param num_concepts: The number of relevant concepts to retrieve literature for.
        :param num_items_per_concept: The number of literature items to retrieve for each concept.
        """
        rag = LanceRAG()
        knowledge_collector = CoreKnowledgeCollector()
        relevant_concepts = self._get_relevant_concepts(num_concepts)
        logger.info('Concepts retrieved: %s', relevant_concepts)

        relevant_concepts.append(self.book_name)

        for concept in relevant_concepts:
            entity_urls = knowledge_collector.get_literature_by_keywords_search([concept], limit=num_items_per_concept)
            for entity_url in entity_urls:
                doc = rag.parse_document(entity_url)
                chunks = rag.chunk_document(doc)
                rag.embed_and_insert_chunks(self.book_name, chunks)
                logger.info('Document inserted into Vector Database: %s', entity_url)

    def _get_relevant_concepts(self, chapter: str | None = None, num_concepts: int = 3) -> List[str]:
        """
        Extract the most relevant concepts from the book.

        :param chapter: The chapter to get relevant concepts for. If None, get concepts for the entire book.
        :param num_concepts: The number of concepts to return.
        """
        concept_extractor_agent = Agent(
            name='ConceptExtractor',
            instructions=CONCEPT_EXTRACTOR_SYSTEM_PROMPT,
            model=self.model,
            output_type=ConceptExtractor,
        )
        input_dict = {
            'book': self.book_name,
            'author': self.author,
            'chapter': chapter,
            'num_concepts': num_concepts,
        }
        concept_extractor_result = Runner.run_sync(concept_extractor_agent, str(input_dict))
        return concept_extractor_result.final_output.concepts

    def _construct_queries_to_search_knowledge_base(self, concepts: List[str]) -> List[str]:
        """
        Construct queries to search the knowledge base using provided concepts.

        :param concepts: The concepts to use to construct the queries.
        """
        query_construction_agent = Agent(
            name='QueryConstructor',
            instructions=QUERY_CONSTRUCTION_SYSTEM_PROMPT,
            model=self.model,
            output_type=VectorDatabaseQuery,
        )
        query_construction_result = Runner.run_sync(query_construction_agent, str(concepts))
        return query_construction_result.final_output.queries

    @staticmethod
    @function_tool()
    def get_relevant_info_from_knowledge_base(book_name: str, queries: List[str]) -> str:
        """Get relevant information from the knowledge base.

        Args:
            book_name: The name of the book to get relevant information for.
            queries: The list of queries to search for in the knowledge base.
        """
        rag = LanceRAG()
        logger.info('Searching for %s in the knowledge base for %s', queries, book_name)
        return '\n'.join(rag.search_table(book_name, queries))
