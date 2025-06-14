import os
from typing import List
from urllib.parse import urlparse, urlunparse

import httpx

from reading_ai_companion.knowledge_collector.base import KnowledgeCollector


class CoreKnowledgeCollector(KnowledgeCollector):
    """
    The implementation of the KnowledgeCollector for the CORE API:
    https://api.core.ac.uk/docs/v3
    """

    def __init__(self):
        assert os.getenv('CORE_API_KEY'), 'CORE_API_KEY is not set'
        self.api_key = os.getenv('CORE_API_KEY')
        self.base_url = 'https://api.core.ac.uk/v3'

    def get_literature_by_keywords_search(self, keywords: List[str], limit: int = 5, offset: int = 0) -> List[str]:
        """
        Get relevant literature for a work based on its title and author.

        Returns a list URLs where the work can be downloaded.
        """
        url = f'{self.base_url}/search/works'
        query = ' AND '.join([f'abstract:{k}' for k in keywords])
        query += ' AND _exists_:downloadUrl'
        params = {'q': query, 'limit': limit, 'offset': offset}

        final_results = []
        with httpx.Client(headers={'Authorization': f'Bearer {self.api_key}'}) as client:
            res = client.post(url, json=params)

            for result in res.json()['results']:
                final_results.append(self._convert_arxiv_url(result['downloadUrl']))
        return final_results

    @staticmethod
    def _convert_arxiv_url(url: str) -> str:
        """
        Convert the arXiv URL to the PDF URL.

        CORE API returns the arXiv Download URL in the form:
        https://arxiv.org/abs/1234.5678

        We need to convert it to the actual download URL in the form:
        https://arxiv.org/pdf/1234.5678
        """
        parsed = urlparse(url)

        # Check if domain contains 'arxiv.org' and path has '/abs/'
        if 'arxiv.org' in parsed.netloc and '/abs/' in parsed.path:
            # Replace '/abs/' with '/pdf/'
            new_path = parsed.path.replace('/abs/', '/pdf/', 1)
            # Rebuild the URL with the new path
            updated_url = urlunparse(parsed._replace(path=new_path))
            return updated_url
        return url  # return original if not matching criteria
