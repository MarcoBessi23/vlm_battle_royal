'''Base class for Vision-Language Models (VLLMs).'''
from abc import ABC, abstractmethod
from PIL import Image

class BaseVLM(ABC):
    """Abstract base class for Vision-Language Models."""

    @abstractmethod
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text output given an image and prompt."""

    @abstractmethod
    def generate_batch(self, batch_size: int, max_new_tokens: int = 1024) -> str:
        """Generate text outputs from batch of images and prompts"""