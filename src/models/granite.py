'''Implementation of Vision-Language Model (VLLM) using IBM Granite Vision.'''
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
from src.models.base_vllm import BaseVLLM

class GraniteVLLM(BaseVLLM):
    """Implementation of BaseVLLM using IBM Granite Vision model."""
    def __init__(self, model_name="ibm-granite/granite-vision-3.2-2b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name).to(self.device)

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)
