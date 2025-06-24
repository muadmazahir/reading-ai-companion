import os
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, apply_chat_template


class Finetuner:
    """
    Class to handle finetuning of models.
    """

    def __init__(self):
        self.lora_model = None
        self._base_model = None
        self._tokenizer = None
        self._base_model_name = None
        self._model_kwargs = None

    @property
    def tokenizer(self):
        """Lazy initialization of tokenizer."""
        if self._tokenizer is None:
            if self._base_model_name is None:
                raise ValueError('base_model_name must be set before accessing tokenizer')
            self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        return self._tokenizer

    @property
    def base_model(self):
        """Lazy initialization of base model."""
        if self._base_model is None:
            if self._base_model_name is None:
                raise ValueError('base_model_name must be set before accessing base_model')
            # Note: You'll need to pass model_kwargs when setting up
            self._base_model = AutoModelForCausalLM.from_pretrained(self._base_model_name, **self._model_kwargs)
        return self._base_model

    def load_dataset(self, dataset_name: str):
        """
        Load a dataset from HuggingFace Hub.

        :param dataset_name: Name of the dataset to load without the username.
        """
        hf_token = self._validate_huggingface_token()
        hf_api = HfApi()
        username = hf_api.whoami()['name']
        try:
            return load_dataset(path=username + '/' + dataset_name, token=hf_token)
        except DatasetNotFoundError:
            raise ValueError(f'Dataset {dataset_name} not found on Hugging Face. Please ensure the dataset is set up')

    def setup_dataset(
        self,
        base_model_name: str,
        data: List[List[str]],
        push_to_hub: bool = False,
        dataset_name: str | None = None,
    ):
        """
        Set up the dataset for fine-tuning.

        :param base_model_name: Name of the base model to use for tokenization.
        :param data: List of [prompt, completion] pairs to create a dataset.
                    Each inner list should contain exactly 2 strings: [prompt, completion].
        :param push_to_hub: Whether to push the created dataset to HuggingFace.
        :param dataset_name: Name of the dataset to push to HuggingFace.
                    Required when push_to_hub is True.

        Eg data:
        [
            ["Who wrote Harry Potter?", "J.K. Rowling"],
            ["Who wrote The Lord of the Rings?", "J.R.R. Tolkien"],
        ]
        """
        self._base_model_name = base_model_name

        prompts = [[{'role': 'user', 'content': row[0]}] for row in data]
        completions = [[{'role': 'assistant', 'content': row[1]}] for row in data]

        dataset_dict = {'prompt': prompts, 'completion': completions}

        untemplated_dataset = Dataset.from_dict(dataset_dict)

        templated_dataset = untemplated_dataset.map(apply_chat_template, fn_kwargs={'tokenizer': self.tokenizer})
        dataset = templated_dataset.train_test_split(test_size=0.25, seed=42)

        if push_to_hub:
            hf_token = self._validate_huggingface_token()
            if dataset_name is None:
                raise ValueError('dataset_name is required when push_to_hub is True')
            dataset.push_to_hub(repo_id=dataset_name, private=True, token=hf_token)
        return dataset

    def setup_lora(self, base_model_name: str, model_kwargs: Dict[str, Any], lora_config: Dict[str, Any]):
        """
        Set up the LoRA model.

        :param base_model_name: Name of the base model to use for tokenization.
        :param model_kwargs: Keyword arguments to use when retrieving the base model.
        :param lora_config: Keyword arguments to use when configuring the LoRA model.
        """
        self._base_model_name = base_model_name
        self._model_kwargs = model_kwargs

        lora_config_obj = LoraConfig(**lora_config)
        self.lora_model = get_peft_model(self.base_model, lora_config_obj)

    def train(self, dataset: Dataset, training_args: Dict[str, Any]):
        """
        Train the LoRA model.

        :param dataset: Dataset to train on.
        :param training_args: Keyword arguments to configure the training.
        """
        self._validate_lora_model()

        # Split dataset into train and eval
        data_train = dataset['train']
        data_eval = dataset['test']

        training_args = SFTConfig(**training_args)

        # Initialize trainer with LoRA
        trainer = SFTTrainer(
            model=self.lora_model,
            args=training_args,
            peft_config=self.lora_model.peft_config['default'],
            processing_class=self.tokenizer,
            train_dataset=data_train,
            eval_dataset=data_eval,
        )

        # Run training
        trainer.train()

    def push_lora_adapter_and_model(self, model_id: str):
        """
        Push the LoRA adapter, model, and tokenizer to Hugging Face.

        :param model_id: ID to give to the model to push to Hugging Face.
        """
        self._validate_lora_model()

        hf_token = self._validate_huggingface_token()

        # push the lora adapters to a separate repository
        self.lora_model.push_to_hub(
            repo_id=model_id + '-lora',
            commit_message='Updated LoRA model',
            token=hf_token,
            save_adapters=True,
            private=True,
        )

        # merge lora adapters back onto the base model using the `merge_and_unload` function
        merged_model = self.lora_model.merge_and_unload()

        # push the merged model to a private HF repository
        merged_model.push_to_hub(
            repo_id=model_id,
            commit_message='Updated SFT model (Base + LoRA)',
            token=hf_token,
            private=True,
        )

        # make sure to also push the tokenizer to the same HF repository
        # We can be sure that the tokenizer is set up because we have already validated the lora model
        # and it is set when lora model is set up
        self.tokenizer.push_to_hub(repo_id=model_id, commit_message='Updated tokenizer', token=hf_token, private=True)

    @staticmethod
    def _validate_huggingface_token():
        """
        Check if the Hugging Face token is set.
        """
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token is None:
            raise ValueError('HF_TOKEN is required. Please set the HF_TOKEN environment variable.')
        return hf_token

    def _validate_lora_model(self):
        """
        Check if the LoRA model is set up.
        """
        if self.lora_model is None:
            raise ValueError('Please ensure lora model is set up using the setup_lora() method')
