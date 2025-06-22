# Reading AI Companion

An intelligent companion that helps you understand books better by providing explanations and answering questions using AI and RAG (Retrieval-Augmented Generation).

## Features

- Create AI companions for any book
- Get AI-powered explanations of book content
- Ask specific questions about the book
- Enhanced understanding through RAG-powered knowledge base
- Extensible architecture for different vector databases and knowledge sources

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/reading-ai-companion.git
cd reading-ai-companion

# Install dependencies using Poetry
poetry install
```

## Environment Setup

You'll need to set up the following environment variables:

### Option 1: Using OpenAI API (Default)

```bash
# Set to use OpenAI API (default behavior)
export USE_OPENAI_API='true'

# OpenAI API key for the AI companion
export LLM_API_KEY='your-openai-api-key'
```

### Option 2: Using separate LLM API

You can use a separate LLM server like local Ollama:

```bash
# Set to use local LLM server
export USE_OPENAI_API='false'

# URL of your LLM server (e.g., Ollama, LM Studio, etc.)
export LLM_API_URL='http://localhost:11434'

# Model name for your local LLM
export LLM_MODEL='gemma3:1b'

# Optional: API key if your server requires authentication
export LLM_API_KEY='your-local-api-key'
```


## Usage

### Basic Usage

```python
from reading_ai_companion.companion import Companion

# Create a companion for a book
companion = Companion('The Wealth of Nations')

# Get an explanation of the book
companion.explain()

# Ask specific questions
companion.query('What was the context in which the book was written?')
```

### Using Knowledge Base

The companion can be enhanced with a knowledge base that provides additional context and information:

```python
# Set up the knowledge base for the book
companion.setup_knowledge_base()

# Get explanations with knowledge base context
companion.explain(use_knowledge_base=True)
```

## Architecture

### Vector Database
- Currently uses LanceDB as the vector database
- Designed to be easily extensible to support other vector databases

### Knowledge Sources
- Currently integrates with CORE API (https://api.core.ac.uk/docs/v3) for academic literature
- Architecture allows for easy addition of new knowledge sources

