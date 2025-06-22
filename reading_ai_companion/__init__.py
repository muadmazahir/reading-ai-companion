import logging
import os

from agents import set_default_openai_api, set_default_openai_client, set_tracing_disabled
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If USE_OPENAI_API is not set, set as default to use the provided LLM server
if os.getenv('USE_OPENAI_API', 'true').lower() in ('0', 'false', 'no', 'off'):
    # Validate required environment variables for non-OpenAI setup
    llm_api_url = os.environ.get('LLM_API_URL')
    if not llm_api_url:
        raise ValueError('LLM_API_URL needs to be set if not using OpenAI API')
    if not os.environ.get('LLM_MODEL'):
        raise ValueError('LLM_MODEL needs to be set if not using OpenAI API')

    # for local LLM server, we might not need a key but a placeholder value is required when instantiating the client
    custom_client = AsyncOpenAI(base_url=llm_api_url, api_key=os.environ.get('LLM_API_KEY', 'placeholder-key'))
    set_default_openai_client(custom_client)
    set_tracing_disabled(True)
    set_default_openai_api('chat_completions')
    logger.info('Using LLM server at %s', llm_api_url)
else:
    llm_api_key = os.environ.get('LLM_API_KEY')
    if not llm_api_key:
        raise ValueError('LLM_API_KEY needs to be set if using OpenAI API')
    os.environ['OPENAI_API_KEY'] = llm_api_key
    logger.info('Using OpenAI API')
