import torch
import torch.nn as nn
from transformers import SiglipProcessor, SiglipModel
from typing import List
from PIL import Image

class FrozenSiglipEncoder(nn.Module):
    def __init__(self, ckpt: str = "google/siglip-so400m-patch14-384", device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        # Load SigLIP model and processor
        self.model = SiglipModel.from_pretrained(ckpt).to(device).eval()
        self.processor = SiglipProcessor.from_pretrained(ckpt)
        
        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad_(False)

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Convert texts to SigLIP embeddings.
        Returns: (B, L) tensor, where L depends on the model variant (e.g., 768 or 1152)
        """
        inputs = self.processor(text=texts, 
                                return_tensors="pt", 
                                padding="max_length", 
                                max_length=64,
                                truncation=True).to(self.device)
        with torch.no_grad():
            # SigLIP uses get_text_features just like CLIP
            text_embeds = self.model.get_text_features(**inputs)
        return text_embeds

    def encode_image(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Convert PIL images to SigLIP embeddings.
        """
        inputs = self.processor.image_processor(
                                            images=images, 
                                            return_tensors="pt", 
                                            do_rescale=False  
                                        ).to(self.device)
        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
        return image_embeds