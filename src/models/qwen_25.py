"""Implementation of BaseVLM using Qwen2.5-VL model."""
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from base_vlm import BaseVLM

class Qwen25VLM(BaseVLM):
    """Implementation using Qwen2.5-VL model."""

    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> str:
        # Qwen2.5-VL uses a different message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)

        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Trim input tokens from output
        output_ids = output_ids[:, inputs.input_ids.shape[1]:]
        
        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def generate_batch(self, images: list[Image.Image], prompt: str, 
                       max_new_tokens: int = 256) -> list[str]:
        # Create messages for each image
        messages_batch = [
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

        # Process all messages
        texts = []
        all_image_inputs = []
        
        for messages in messages_batch:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

            image_inputs, _ = process_vision_info(messages)
            all_image_inputs.extend(image_inputs)

        # Prepare batched inputs
        inputs = self.processor(
            text=texts,
            images=all_image_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)

        # Generate
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Trim input tokens from output
        output_ids = output_ids[:, inputs.input_ids.shape[1]:]

        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
