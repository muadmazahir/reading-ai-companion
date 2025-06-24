# Reading AI Companion

An intelligent companion that helps you understand books better by providing explanations and answering questions using various AI techniques like RAG and finetuning.

## Features

- Create AI companions for any book
- Get AI-powered explanations of book content
- Ask specific questions about the book
- Enhanced understanding through RAG-powered knowledge base
- Fine-tune open-source LLMs to be better at answering questions about specific books
- Extensible architecture for different vector databases, knowledge sources, and underlying base LLM models

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

### Fine-tuning Setup

To use the fine-tuning functionality, you'll need a Hugging Face account and token:

```bash
# Hugging Face token for uploading datasets and models
export HF_TOKEN='your-huggingface-token'
```

You can get your Hugging Face token [here](https://huggingface.co/settings/tokens).


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

### Fine-tuning for specialised models

You can fine-tune an open-source LLM to be better at answering questions about a specific book. This process involves creating a synthetic dataset from your RAG knowledge base and then training the model.

#### Step 1: Create Training Dataset

First, ensure you have set up your knowledge base, then create a dataset for fine-tuning:

```python
# Set up the knowledge base first (if not already done)
companion.setup_knowledge_base()

# Create a dataset for fine-tuning
companion.setup_dataset_for_finetuning(base_model_name='Qwen/Qwen2.5-0.5B-Instruct')
```

The agent will automatically:
- Extract relevant concepts from the book
- Use your RAG knowledge base to provide context
- Generate synthetic prompt-completion pairs for training
- Upload the dataset to Hugging Face

#### Step 2: Configure Training

Create a configuration file for the fine-tuning process and set your required configuration. You can use the provided `config-example.json` as a template.

By default, the configuration file should be located at `./config.json`. You can place it elsewhere and point to it using the `CONFIG_FILE_PATH` environment variable:

```bash
export CONFIG_FILE_PATH='/path/to/your/config.json'
```

#### Step 3: Fine-tune the Model

Once you have your dataset and configuration ready, you can fine-tune the model:

```python
# Fine-tune the model on your dataset
companion.finetune(base_model_name='Qwen/Qwen2.5-0.5B-Instruct')
```

The fine-tuned model will be automatically uploaded to Hugging Face with a name like `Qwen2.5-0.5B-Instruct-The-Great-Gatsby`.


## Architecture

### Vector Database
- Currently uses LanceDB as the vector database
- Designed to be easily extensible to support other vector databases

### Knowledge Sources
- Currently integrates with CORE API (https://api.core.ac.uk/docs/v3) for academic literature
- Architecture allows for easy addition of new knowledge sources

### Fine-tuning Pipeline
- Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Generates synthetic training data using RAG knowledge base
- Supports various open-source models from Hugging Face
- Automatically handles dataset and model upload to Hugging Face
