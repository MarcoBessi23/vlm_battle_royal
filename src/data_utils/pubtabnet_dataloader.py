'''Data loader for PubTabNet dataset.'''
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL import Image
from datasets import Dataset
import torch
from torch.utils.data import DataLoader

class PubTabNetLoader:
    """loader for PubTabNet images with HF VLM support."""
    
    SUPPORTED_EXT = {".png", ".jpg", ".jpeg"}
    
    def __init__(self, data_dir: str, max_size: Optional[int] = None):
        """
        Args:
            data_dir: Directory containing PubTabNet images
            max_size: Optional max image dimension (resizes if larger)
        """
        self.data_dir = Path(data_dir)
        self.max_size = max_size
        self.image_paths = self._scan_images()
        
    def _scan_images(self) -> List[Path]:
        """Scan directory for valid images."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
            
        paths = [
            p for p in self.data_dir.iterdir()
            if p.suffix.lower() in self.SUPPORTED_EXT
        ]
        print(f"Found {len(paths)} images in {self.data_dir}")
        return sorted(paths)
    
    def load_image(self, img_path: Path) -> Optional[Image.Image]:
        """Load and validate a single image."""
        try:
            img = Image.open(img_path)
            img.load()  # Force loading to catch truncated images
            img = img.convert("RGB")
            
            # Optional resizing for memory efficiency
            if self.max_size and max(img.size) > self.max_size:
                img.thumbnail((self.max_size, self.max_size), Image.Resampling.LANCZOS)
            
            return img
        except Exception as e:
            print(f" Failed to load {img_path.name}: {e}")
            return None
    
    def to_hf_dataset(self) -> Dataset:
        """Create Hugging Face Dataset """
        def generator():
            for path in self.image_paths:
                img = self.load_image(path)
                if img is not None:
                    yield {
                        "image": img,
                        "image_path": str(path),
                        "filename": path.name
                    }
        
        dataset = Dataset.from_generator(generator)
        print(f" Created HF dataset with {len(dataset)} valid images")
        return dataset

