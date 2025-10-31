from datasets import load_dataset
from PIL import Image
from vllm import LLM, SamplingParams
from pathlib import Path
import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run Granite vLLM on PubTabNet")
    parser.add_argument("--data_dir", type=str, default="data/PubTabNet_subset")
    parser.add_argument("--save_path", type=str, default="results/inference.jsonl")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of images to process at once")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results file")
    return parser.parse_args()

class Granite_vllm:
    def __init__(self, model_name="ibm-granite/granite-vision-3.2-2b"):
        self.model = LLM(model=model_name, limit_mm_per_prompt={"image": 1})
        self.sampling_params = SamplingParams(temperature=0.2, max_tokens=512)
        self.system_prompt = (
            "<|system|>\n"
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        )
        self.question = "Task: Look carefully at the table in the image. Recreate the same table structure in valid HTML."
        self.image_token = "<image>"
    
    def build_prompt(self):
        return f"{self.system_prompt}<|user|>\n{self.image_token}\n{self.question}\n<|assistant|>\n"
    
    def generate_batch(self, images):
        """
        images: list of PIL.Image
        Returns: list of vLLM outputs
        """
        prompt = self.build_prompt()
        inputs = []
        
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img},
            })
        
        outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
        return outputs

def main():
    args = parse_args()
    
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading Granite Vision model...")
    granite = Granite_vllm()
    print("loading dataset..") 
    dataset = load_dataset("imagefolder", data_dir=args.data_dir, split="test")
    
    with open(args.save_path, "w") as f:
        for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
            batch_end = min(i + args.batch_size, len(dataset))
            batch_images = []
            batch_metadata = []
            for j in range(i, batch_end):
                data = dataset[j]  
                sample_id = str(j)
                if hasattr(data["image"], "filename") and data["image"].filename:
                    sample_id = data["image"].filename
                
                img = data["image"]
                batch_images.append(img)
                batch_metadata.append({
                    "id": sample_id,
                })
            
                results = granite.generate_batch(batch_images)
                
            for meta, result in zip(batch_metadata, results):
                output_data = {
                    "id": meta["id"],
                    "output": result.outputs[0].text
                }
                json.dump(output_data, f)
                f.write("\n")
                f.flush() 

    print(f"Saved results to {args.save_path}")

if __name__ == "__main__":
    main() 
