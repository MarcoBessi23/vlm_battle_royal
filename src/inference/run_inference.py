'''Script to run inference using a specified Vision-Language Model (VLLM) on PubTabNet images.'''
import argparse
import json
from tqdm import tqdm
from src.data.pubtabnet_dataloader import load_pubtabnet_images, load_image
from src.models.load_model import create_model

def parse_args():
    '''Parse command-line arguments.'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="Model name to use for inference."
                        )
    parser.add_argument("--data_dir",
                        type=str,
                        default="data/PubTabNet_subset",
                        help="Directory containing PubTabNet images."
                        )
    parser.add_argument("--prompt",
                        type=str,
                        default="Extract the table structure.",
                        help="Prompt to guide the model."
                        )
    parser.add_argument("--save_path",
                        type=str,
                        default="experiments/results/inference.json",
                        help="Path to save inference results."
                        )
    return parser.parse_args()

def run_inference(model_name, data_dir, prompt, save_path):
    '''Run inference on images in the specified directory using the given model and prompt.'''
    model = create_model(model_name)
    img_paths = load_pubtabnet_images(data_dir)
    results = []

    for img_path in tqdm(img_paths, desc=f"Running {model_name}"):
        image = load_image(img_path)
        output = model.generate(image, prompt)
        results.append({"image": img_path, "output": output})

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model, args.data_dir, args.prompt, args.save_path)
