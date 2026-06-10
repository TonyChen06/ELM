"""HF vision encoders behind one interface: processor features -> token embeddings."""
import torch
from torch import nn
from transformers import AutoModel


class HFVision(nn.Module):
    def __init__(self, hf_id):
        super().__init__()
        self.model = AutoModel.from_pretrained(hf_id)
        self.embed_dim = getattr(self.model.config, "projection_dim", None) \
            or getattr(getattr(self.model.config, "vision_config", self.model.config), "hidden_size")

    @torch.no_grad()
    def forward(self, **features):
        if hasattr(self.model, "get_image_features"):  # CLIP, SigLIP2
            out = self.model.get_image_features(**features)
            if not isinstance(out, torch.Tensor):  # transformers 5.x returns an output object
                out = out.pooler_output
            return out.unsqueeze(1) if out.ndim == 2 else out
        out = self.model(**features).last_hidden_state  # plain ViT
        return out.mean(dim=1, keepdim=True)
