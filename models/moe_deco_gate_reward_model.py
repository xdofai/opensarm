import torch
import torch.nn as nn
import torch.nn.functional as F


class _MoEExpert(nn.Module):
    """A lightweight expert that maps a per-time fused feature to a scalar logit."""
    def __init__(self, in_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hd = in_dim if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hd),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hd, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (M, in_dim) -> (M, 1)
        return self.net(x)


# --- helpers ---
def _act(name: str):
    name = name.lower()
    return {"gelu": nn.GELU(), "relu": nn.ReLU(), "elu": nn.ELU()}[name]


def _make_mlp(in_dim: int,
              units: list[int],
              out_dim: int,
              act: str = "gelu",
              dropout: float = 0.0,
              layernorm_first: bool = True) -> nn.Sequential:
    """Generic MLP builder: [LayerNorm?] -> (Linear+Act+Dropout)* -> Linear(out_dim)."""
    layers = []
    if layernorm_first:
        layers.append(nn.LayerNorm(in_dim))
    prev = in_dim
    for u in units:
        layers += [nn.Linear(prev, u), _act(act)]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        prev = u
    layers += [nn.Linear(prev, out_dim)]
    return nn.Sequential(*layers)


class RewardTransformer(nn.Module):
    def __init__(self,
                 d_model: int = 512,
                 vis_emb_dim: int = 512,
                 text_emb_dim: int = 512,          # encoder output dim
                 state_dim: int = 14,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 num_cameras: int = 1,
                 # === Multi-gate MoE ===
                 num_gates: int = 3,
                 # === MoE hyper-parameters ===
                 num_experts: int = 8,
                 top_k: int = 2,
                 gate_units: list[int] | None = [512, 256],
                 gate_act: str = "gelu",
                 gate_dropout: float = 0.0,
                 expert_units: list[int] | None = [512, 512, 512],
                 expert_act: str = "gelu",
                 expert_dropout: float = 0.0,
                 lambda_balance: float = 5.0,
                 lambda_entropy: float = 0.1,
                 lambda_importance: float = 0.1
                 ):
        super().__init__()
        self.d_model = d_model
        self.text_emb_dim = text_emb_dim
        self.num_cameras = num_cameras

        # MoE configuration
        self.num_experts = num_experts
        self.top_k = top_k
        self.lambda_balance = lambda_balance
        self.lambda_entropy = lambda_entropy
        self.lambda_importance = lambda_importance

        # Multi-gate configuration: one gate network per class; the active gate is
        # chosen per-sample by the externally-supplied gate_idx.
        self.num_gates = num_gates
        
        # Projection layers
        self.visual_proj = nn.Linear(vis_emb_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)

        # Language projection is ONLY for feeding the transformer (maps encoder dim -> d_model).
        self.lang_proj = nn.Identity() if (text_emb_dim == d_model) else nn.Linear(text_emb_dim, d_model)
        self.task_inst_proj = nn.Identity() if (text_emb_dim == d_model) else nn.Linear(text_emb_dim, d_model)


        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

        # Positional bias only for first visual frame (to avoid leaking absolute time)
        self.first_pos = nn.Parameter(torch.zeros(1, d_model))

        # ======= MoE components (multi-gate) =======
        # Input to gate/experts is the per-time fused feature of shape (B, T, (N+3)*d_model)
        # (N camera tokens + lang + task_inst + state)
        self.moe_in_dim = d_model * (num_cameras + 3)

        # Resolve defaults that depend on moe_in_dim
        if gate_units is None:
            gate_units = [512]
        if expert_units is None:
            expert_units = [self.moe_in_dim]

        # Multi-gate: each gate has its own gating network (in MoE input space)
        self.gate_nets = nn.ModuleList([
            _make_mlp(
                self.moe_in_dim,
                units=gate_units,
                out_dim=self.num_experts,
                act=gate_act,
                dropout=gate_dropout,
                layernorm_first=True
            )
            for _ in range(self.num_gates)
        ])

        # Experts: shared across all gates
        self.experts = nn.ModuleList([
            _make_mlp(
                self.moe_in_dim,
                units=expert_units,
                out_dim=1,
                act=expert_act,
                dropout=expert_dropout,
                layernorm_first=True
            )
            for _ in range(self.num_experts)
        ])

        # Buffers to expose auxiliary info after forward
        self.last_aux_loss: torch.Tensor | None = None
        self.last_gate_entropy: float | None = None
        self.last_expert_load: torch.Tensor | None = None  # normalized frequency over experts

        # Sanity checks
        if not (1 <= self.top_k <= self.num_experts):
            raise ValueError(
                f"top_k must be in [1, num_experts], got {self.top_k} vs {self.num_experts}"
            )

    def forward(self,
                img_seq: torch.Tensor,          # (B, N, T, vis_emb_dim)
                lang_emb: torch.Tensor,         # encoder output: (B, text_emb_dim)
                task_inst_emb: torch.Tensor,    # (B, text_emb_dim)
                state: torch.Tensor,            # (B, T, state_dim)
                lengths: torch.Tensor,          # (B,)
                gate_idx: torch.Tensor | None = None,  # (B,) optional externally-supplied gate index
                ):
        """
        Returns:
            r: (B, T) in [0, 1]; progress per time step.
            aux_loss: scalar MoE auxiliary loss.
            info: dict with gate_entropy, expert_load, and importance_loss.
        Gating:
            - `gate_idx` (per-sample class index) selects which gate network to use;
              it is required (prototype-based selection is not implemented).
        Transformer:
            - Uses lang_proj(lang_emb) in d_model space.
        """
        B, N, T, _ = img_seq.shape  # N = num_cameras
        D = self.d_model
        device = img_seq.device
        eps = 1e-9

        # === Project vision ===
        vis_proj = self.visual_proj(img_seq)                           # (B, N, T, d_model)

        # === Project state ===
        state_proj = self.state_proj(state).unsqueeze(1)               # (B, 1, T, d_model)

        # === Language: project (or identity) then broadcast over time for the transformer ===
        lang_token_proj = self.lang_proj(lang_emb)                 # (B, d_model)
        lang_proj = lang_token_proj.unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)            # (B, 1, T, d_model)
        
        task_inst_token_proj = self.task_inst_proj(task_inst_emb)  # (B,D)
        task_inst_proj = task_inst_token_proj.unsqueeze(1).unsqueeze(2).expand(B, 1, T, D)  # (B,1,T,D)


        # === Multi-gate selection ===
        if gate_idx is not None:
            if gate_idx.dim() != 1 or gate_idx.size(0) != B:
                raise ValueError(
                    f"gate_idx must have shape (B,) with B={B}, got {tuple(gate_idx.shape)}"
                )
            gate_idx_per_sample = gate_idx.to(device=device, dtype=torch.long)
        else:
            raise NotImplementedError("Prototype-based gate selection is not implemented in this version.")
        
        # === Concatenate tokens along the "camera/state/lang" axis ===
        x = torch.cat([vis_proj, lang_proj, task_inst_proj, state_proj], dim=1)        # (B, N+3, T, d_model)

        # === Add positional bias to the first visual frame of each camera ===
        x[:, :N, 0, :] += self.first_pos                               # (B, N, T, d_model)

        # === Reshape for transformer: (B, (N+3)*T, d_model) ===
        x = x.view(B, (N + 3) * T, D)
        L = x.size(1)

        # === Build key padding mask for transformer ===
        base_mask = torch.arange(T, device=device).expand(B, T) >= lengths.unsqueeze(1)  # (B, T)
        mask = base_mask.unsqueeze(1).expand(B, N + 3, T).reshape(B, (N + 3) * T)        # (B, L)

        # --- Dynamic causal mask (upper-triangular) for autoregressive visibility ---
        causal_mask = torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1
        )  # (L, L)

        # --- Transformer encoding with causal attention and padding mask ---
        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=mask,
            is_causal=True
        )  # (B, L, d_model)

        # === Reshape back to (B, N+3, T, d_model) ===
        h = h.view(B, N + 3, T, D)

        # === Fuse per time step features by concatenation: (B, T, moe_in_dim) ===
        fused = h.permute(0, 2, 1, 3).contiguous().view(B, T, -1)      # (B, T, moe_in_dim)

        # === Valid time-step mask (exclude padded positions from MoE + losses) ===
        valid_mask = ~base_mask                                        # (B, T) bool, True = valid
        BT = B * T
        fused_flat = fused.view(BT, self.moe_in_dim)                   # (B*T, moe_in_dim)
        valid_idx = valid_mask.view(-1).nonzero(as_tuple=False).squeeze(1)  # (Nv,)
        Nv = valid_idx.numel()

        # If no valid tokens (edge case), return zeros and zero aux
        if Nv == 0:
            r = torch.zeros(B, T, device=device)
            self.last_aux_loss = torch.zeros((), device=device)
            self.last_gate_entropy = 0.0
            self.last_expert_load = torch.zeros(self.num_experts, device=device)
            return torch.sigmoid(r), self.last_aux_loss, {
                "gate_entropy": self.last_gate_entropy,
                "expert_load": self.last_expert_load,
            }

        # === Prepare per-token gate index (which gate net to use) ===
        b_idx = (valid_idx // T).to(device)                            # (Nv,)
        token_gate_idx = gate_idx_per_sample[b_idx]                    # (Nv,) in [0, num_gates-1]

        # === Gate network on valid tokens only (multi-gate) ===
        gate_in = fused_flat[valid_idx]                                # (Nv, moe_in_dim)
        gate_logits = torch.empty(Nv, self.num_experts, device=device) # (Nv, E)

        for g in range(self.num_gates):
            mask_g = (token_gate_idx == g)
            if mask_g.any():
                rows = mask_g.nonzero(as_tuple=False).squeeze(1)       # (n_g,)
                gate_logits[rows] = self.gate_nets[g](gate_in[rows])   # (n_g, E)

        gate_probs = F.softmax(gate_logits, dim=-1)                    # (Nv, E)

        # === Top-k sparse routing ===
        topk_scores, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)  # (Nv, k)

        # === Run only the selected experts (sparse execution) ===
        selected_outputs = torch.zeros(Nv, self.top_k, 1, device=device)  # (Nv, k, 1)

        unique_experts = torch.unique(topk_indices)
        for e in unique_experts.tolist():
            row_mask = (topk_indices == e).any(dim=-1)                 # (Nv,)
            if not row_mask.any():
                continue
            rows = row_mask.nonzero(as_tuple=False).squeeze(1)         # (n_e,)
            x_e = gate_in[rows]                                        # (n_e, moe_in_dim)
            y_e = self.experts[e](x_e)                                 # (n_e, 1)

            slot_mask = (topk_indices[rows] == e)                      # (n_e, k)
            y_expand = y_e.unsqueeze(1).expand(-1, self.top_k, -1)     # (n_e, k, 1)
            selected_outputs[rows] = torch.where(
                slot_mask.unsqueeze(-1),
                y_expand,
                selected_outputs[rows]
            )

        # === Weighted fusion across the k selected experts ===
        weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + eps)  # (Nv, k)
        y_valid = (selected_outputs.squeeze(-1) * weights).sum(dim=-1, keepdim=True)  # (Nv, 1)

        # === Scatter y_valid back to (B, T) and apply sigmoid to get [0,1] ===
        y_flat = torch.zeros(BT, 1, device=device)
        y_flat[valid_idx] = y_valid
        r = y_flat.view(B, T)  # (B, T)
        r = torch.sigmoid(r)  # map to [0, 1]

        # === Auxiliary losses for MoE routing (computed on valid tokens only) ===
        one_hot = F.one_hot(topk_indices, num_classes=self.num_experts).float()  # (Nv, k, E)
        expert_count = one_hot.sum(dim=(0, 1))                                   # (E,)
        expert_load = expert_count / (float(Nv * self.top_k) + eps)              # (E,)
        load_balancing_loss = expert_load.var(unbiased=False)

        # Entropy regularization: encourage high entropy in gate distributions
        ent = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()
        
        # Importance loss
        importance = gate_probs.sum(dim=0)                      # (E,)
        importance = importance / (importance.sum() + eps)      # normalize
        importance_loss = importance.var(unbiased=False)
        
        aux_loss = (
            self.lambda_balance * load_balancing_loss
            + self.lambda_importance * importance_loss
            - self.lambda_entropy * ent
        )

        # Expose diagnostics
        self.last_aux_loss = aux_loss
        self.last_gate_entropy = float(ent.detach().item())
        self.last_expert_load = expert_load.detach()
        self.last_importance_loss = importance_loss.detach()
        info = {
            "gate_entropy": self.last_gate_entropy,
            "expert_load": self.last_expert_load,
            "importance_loss": self.last_importance_loss,
        }

        return r, aux_loss, info
