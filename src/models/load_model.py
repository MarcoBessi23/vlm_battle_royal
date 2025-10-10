'''Factory function to create and return instances of different VLLM models.'''
from src.models.granite import GraniteVLLM

def create_model(model_name: str):
    """Return an instance of the selected model."""
    model_name = model_name.lower()
    if "granite" in model_name:
        return GraniteVLLM(model_name)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
