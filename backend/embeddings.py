"""
This module defines the `TextEmbedder` class, a Singleton responsible for generating embeddings
from input text using a pre-trained transformer model. The model configuration is specified in `settings.py`.
"""

import logging
import traceback
from pathlib import Path
from threading import Lock
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from backend.settings import AppConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """
    A thread-safe implementation of the Singleton pattern, ensuring that only one instance
    of the class is created across all threads.
    """

    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Overrides the default __call__ method to implement the Singleton pattern.
        Ensures only one instance of the class is created, even in multithreaded environments.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class TextEmbedder(metaclass=SingletonMeta):
    """
    A Singleton class responsible for generating text embeddings using a pre-trained transformer model.
    The transformer model, tokenizer, and device settings are defined in the project's settings file.
    """

    def __init__(
        self,
        config: AppConfig,
        cache_dir: Optional[Path] = None,
        token_limit: int = 256,
    ):
        """
        Initialize the TextEmbedder with model and tokenizer. Loads the model onto the specified device.

        :param model_id: Identifier for the transformer model.
        :param max_input_length: Maximum number of tokens for the input text.
        :param device: The device to run the model on (e.g., 'cpu' or 'cuda').
        :param cache_dir: Optional directory for caching the pre-trained model.
        :param token_limit: Limit for the number of tokens processed in a single call.
        """
        self.config = config
        self._model_id = self.config.get("EMBEDDING_MODEL_ID")
        self._device = self.config.get("EMBEDDING_MODEL_DEVICE")
        self._max_input_length = int(
            self.config.get("EMBEDDING_MODEL_MAX_INPUT_LENGTH")
        )
        self._token_limit = token_limit

        # Load tokenizer and model
        try:
            logger.info(f"Loading tokenizer for model {self._model_id}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            logger.info(f"Loading model {self._model_id} on device {self._device}")
            self._model = AutoModel.from_pretrained(
                self._model_id,
                cache_dir=str(cache_dir) if cache_dir else None,
            ).to(self._device)
            self._model.eval()
        except Exception as e:
            logger.error(f"Error initializing model {self._model_id}: {str(e)}")
            raise RuntimeError(
                f"Failed to load model or tokenizer for {self._model_id}"
            ) from e

    @property
    def token_limit(self) -> int:
        """Returns the token limit used in embedding generation."""
        return self._token_limit

    @property
    def model_id(self) -> str:
        """Returns the model identifier."""
        return self._model_id

    @property
    def max_input_length(self) -> int:
        """Returns the maximum allowed input length for the model."""
        return self._max_input_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Returns the tokenizer used by the model."""
        return self._tokenizer

    def __call__(
        self, input_text: str, to_list: bool = True
    ) -> Union[np.ndarray, List[float]]:
        """
        Generates embeddings for the given input text using the pre-trained model.

        :param input_text: The input text for which embeddings are to be generated.
        :param to_list: Whether to return the embeddings as a flattened list or numpy array.
        :return: A list of embeddings (or numpy array) corresponding to the input text.
        """
        if not input_text:
            logger.warning("Received empty input text.")
            return [] if to_list else np.array([])

        try:
            # Tokenize the input text
            tokenized_text = self._tokenizer(
                input_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_input_length,
            ).to(self._device)
            logger.debug(f"Tokenized text for input: {input_text[:50]}...")

        except Exception:
            logger.error(f"Tokenization error: {traceback.format_exc()}")
            return [] if to_list else np.array([])

        try:
            # Generate embeddings using the transformer model
            with torch.no_grad():  # Disable gradient calculation for efficiency
                result = self._model(**tokenized_text)
            embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
            logger.info(f"Generated embeddings for input: {input_text[:50]}...")

        except Exception:
            logger.error(
                f"Embedding generation error for model {self._model_id}: {traceback.format_exc()}"
            )
            return [] if to_list else np.array([])

        # Return embeddings either as a flattened list or numpy array
        return embeddings.flatten().tolist() if to_list else embeddings
