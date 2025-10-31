"""Script to run inference using a specified Vision-Language Model (VLM) on PubTabNet images."""
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from src.data_utils.pubtabnet_dataloader import PubTabNetLoader 
from src.models.load_model import create_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VLM inference on PubTabNet dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name to use for inference (e.g., 'florence-2', 'qwen-vl')"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/PubTabNet_subset",
        help="Directory containing PubTabNet images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Extract the table structure.",
        help="Prompt to guide the model"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="experiments/results/inference.jsonl",
        help="Path to save inference results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1 for safety)"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Maximum image dimension (resizes if larger)"
    )
    return parser.parse_args()


def run_inference(model_name, data_dir, prompt, save_path, batch_size=1, 
                  max_images=None, max_size=1024):
    """Run inference on images using the specified VLM model.
    
    Args:
        model_name: Name of the VLM model to use
        data_dir: Directory containing images
        prompt: Text prompt for the model
        save_path: Where to save results JSON
        batch_size: Number of images to process at once
        max_images: Optional limit on number of images
        max_size: Maximum image dimension for resizing
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("")
    # Initialize model
    print(f"Loading model: {model_name}")
    model = create_model(model_name)
    
    # Initialize data loader
    print(f"Loading images from: {data_dir}")
    loader = PubTabNetLoader(data_dir, max_size=max_size)
    
    # Create HF dataset
    dataset = loader.to_hf_dataset()
    
    # Optionally limit dataset size
    if max_images is not None:
        dataset = dataset.select(range(min(max_images, len(dataset))))
        print(f"Limited to {len(dataset)} images for testing")
    
    print(f"Processing {len(dataset)} images with prompt: '{prompt}'")
        
    if batch_size == 1:
        # Single image processing (more compatible with most VLMs)
        for item in tqdm(dataset, desc=f"Running {model_name}"):
            image = item["image"]
            output = model.generate(image, prompt)
            
            result = {
                "image_path": item["image_path"],
                "filename": item["filename"],
                "output": output,
                "prompt": prompt
            }
            with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + '\n') 
    else:
        # Batch processing (if model supports it)
        print(f"Using batch size: {batch_size}")
        for i in tqdm(range(0, len(dataset[:100]), batch_size), desc=f"Running {model_name}"):
            batch = dataset[i:i + batch_size]
            
            # Handle both single item and batch
            images = batch["image"] if isinstance(batch["image"], list) else [batch["image"]]
            paths = batch["image_path"] if isinstance(batch["image_path"], list) else [batch["image_path"]]
            filenames = batch["filename"] if isinstance(batch["filename"], list) else [batch["filename"]]
            
            # Generate outputs for batch
            outputs = model.generate_batch(images, prompt)
            assistant_token = "<|assistant|>"
            
            # Store results removing context
            for path, filename, output in zip(paths, filenames, outputs):
                if assistant_token in output:
                    output = output.split(assistant_token, 1)[1].strip()
                result = {
                    "image_path": path,
                    "filename": filename,
                    "output": output,
                    "prompt": prompt
                }
                with open(save_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + '\n')        
            
if __name__ == "__main__":
    args = parse_args() 
    run_inference(
        model_name=args.model_name,
        data_dir=args.data_dir,
        prompt=args.prompt,
        save_path=args.save_path,
        batch_size=args.batch_size,
        max_images=args.max_images,
        max_size=args.max_size
    )


