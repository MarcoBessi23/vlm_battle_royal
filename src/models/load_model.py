"""Factory function to create and return instances of different VLM models."""
from src.models.granite import GraniteVLM

def create_model(model_name: str):
    """
    Factory function to create VLM model instances.
    
    Args:
        model_name: Model identifier (case-insensitive)
                   Examples: "granite", "granite-vision-3.2-2b", 
                            "ibm-granite/granite-vision-3.2-2b"
    
    Returns:
        Model instance with generate() and generate_batch() methods
        
    Raises:
        ValueError: If model_name is not supported
    """
    model_name_lower = model_name.lower()
    
    # Granite Vision models
    if "granite" in model_name_lower:
        print(f"Creating Granite Vision model: {model_name}")
        return GraniteVLM()
    
    else:
        raise ValueError(
            f"Unsupported model: {model_name}\n"
            f"Supported models:\n"
            f"  • granite (Granite Vision 3.2)\n"
            f"  • granite-vision-3.2-2b\n"
            f"  • granite-vision-3.3-2b\n"
            # f"  • florence-2\n"
            # f"  • qwen-vl\n"
        )


# Optional: Helper to list available models
def list_available_models():
    """Return a list of all supported model names."""
    return {
        "granite": [
            "granite",
            "granite-vision",
            "granite-vision-3.2-2b",
            "granite-vision-3.3-2b",
            "ibm-granite/granite-vision-3.2-2b",
        ],
        # "florence": ["florence-2", "microsoft/Florence-2-large"],
        # "qwen": ["qwen-vl", "Qwen/Qwen-VL"],
    }


if __name__ == "__main__":
    # Test the factory
    print("Testing model factory:\n")
    
    test_names = [
        "granite",
        "Granite-Vision-3.2-2B",
        "ibm-granite/granite-vision-3.2-2b",
    ]
    
    for name in test_names:
        print(f"Creating model: {name}")
        try:
            model = create_model(name)
            print(f" model: {type(model).__name__}\n")
        except Exception as e:
            print(f" Not found model: {e}\n")
