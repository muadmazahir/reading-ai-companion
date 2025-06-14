from abc import ABC, abstractmethod
from typing import List


class KnowledgeCollector(ABC):
    """
    Abstract base class for knowledge retrieval systems.
    """

    @abstractmethod
    def get_literature_by_keywords_search(self, keywords: List[str], limit: int = 5, offset: int = 0) -> List[str]:
        """
        Fetch relevant literature download URLs based on keywords.

        :param keywords: The keywords to search for.
        :param limit: The number of results to return.
        :param offset: The offset to start from.
        :return: A list of URLs where the literature can be downloaded.
        """
        pass
