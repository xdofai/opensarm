import torch
import torch.nn as nn
from typing import Optional, Tuple

class ActionTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 vis_emb_dim: int = 512,
                 state_dim: int = 7,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 num_cameras: int = 1,
                 num_tasks: int = 27,
                 num_classes: Optional[int] = None,
                 max_t: int = 200,  # Maximum supported sequence length
                 ):
        super().__init__()
        self.d_model = d_model
        self.num_cameras = num_cameras
        self.max_t = max_t

        # Projections
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # --- Positional Encoding ---
        # Learned positional embeddings for each frame
        self.time_pos_emb = nn.Parameter(torch.zeros(1, 1, max_t, d_model))

        # Encoder Backbone
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # Shared fusion MLP
        fused_in = d_model * (num_cameras + 1)
        self.fusion_backbone = nn.Sequential(
            nn.LayerNorm(fused_in),
            nn.Linear(fused_in, d_model),
            nn.ReLU(),
        )

        self.task_head = nn.Linear(d_model, num_tasks)
        self.class_head = nn.Linear(d_model, num_classes) if num_classes is not None else None

    def forward(self,
                img_seq: torch.Tensor,     # (B, N, T, vis_emb_dim)
                state: torch.Tensor,       # (B, T, state_dim)
                lengths: torch.Tensor,     # (B,)
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, T, _ = img_seq.shape
        D = self.d_model
        device = img_seq.device

        # 1. Input Projections
        vis_proj = self.visual_proj(img_seq)                 # (B, N, T, D)
        state_proj = self.state_proj(state).unsqueeze(1)     # (B, 1, T, D)

        # 2. Token Concatenation
        # x shape: (B, N + 1, T, D)
        x = torch.cat([vis_proj, state_proj], dim=1)         

        # 3. Add Temporal Position Encoding
        # We slice the embedding to current T and broadcast across B and (N+1)
        # self.time_pos_emb[:, :, :T, :] has shape (1, 1, T, D)
        x = x + self.time_pos_emb[:, :, :T, :]

        # 4. Flatten for Transformer
        # (B, (N + 1) * T, D)
        x = x.view(B, (N + 1) * T, D)
        L = x.size(1)
        
        # 5. Masking Logic
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)
        pad_mask = base_mask.unsqueeze(1).expand(B, N + 1, T).reshape(B, (N + 1) * T)
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        
        # 6. Encoding
        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
            is_causal=True
        )
        
        # 7. Spatio-Temporal Fusion
        # Reshape back to (B, T, (N+1)*D) to fuse camera and state tokens per frame
        h = h.view(B, N + 1, T, D).permute(0, 2, 1, 3).reshape(B, T, (N + 1) * D)
        fused = self.fusion_backbone(h) 
        
        # 8. Final State Extraction (Last valid frame)
        last_idx = (lengths - 1).long()
        fused_final = fused[torch.arange(B), last_idx] 

        # 9. Multi-task Output Logic
        task_logits = self.task_head(fused_final)
        
        if self.class_head is None:
            return task_logits
        else:
            class_logits = self.class_head(fused_final)
            return task_logits, class_logits