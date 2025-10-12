"""Implementation of BaseVLM using IBM Granite Vision model."""
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class GraniteVLM:
    """Implementation using IBM Granite Vision model."""
    
    def __init__(self, model_name="ibm-granite/granite-vision-3.2-2b", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        
        # Fix padding side for decoder-only models (required for batching)
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
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
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def generate_batch(self, images: list[Image.Image], prompt: str, 
                       max_new_tokens: int = 256) -> list[str]:
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for img in images
        ]
        
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)
