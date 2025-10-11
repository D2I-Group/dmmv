import torch
from torch import nn
import torch.nn.functional as F
import einops


class DoubleLinearCombine(nn.Module):
    def __init__(
        self, time_feature_num, time_feature_dim, vision_feature_num, vision_feature_dim
    ):
        super().__init__()
        self.dimension_reduction = nn.Linear(vision_feature_dim, time_feature_dim)
        self.embedding_num_reduction = nn.Linear(vision_feature_num, time_feature_num)
        self.combine_head = nn.Linear(time_feature_num * 2, time_feature_num)

    def forward(self, F_time: torch.Tensor, F_vision: torch.Tensor) -> torch.Tensor:
        # F_time: (batch_size,num_variables, time_feature_dim, time_feature_num)
        # F_vision: (batch_size, vision_feature_num, vision_feature_dim)
        F_vision = self.dimension_reduction(F_vision)
        F_vision = einops.rearrange(F_vision, "b n d -> b d n")
        F_vision = self.embedding_num_reduction(F_vision)
        F_vision = einops.rearrange(F_vision, "b d n -> b 1 d n")
        F_vision = F_vision.expand(-1, F_time.shape[1], -1, -1)
        F_fusion = torch.cat([F_time, F_vision], dim=-1)
        F_fusion = self.combine_head(F_fusion)
        return F_fusion


class DirectCombine(nn.Module):
    def __init__(
        self,
        time_feature_num,
        time_feature_dim,
        vision_feature_num,
        vision_feature_dim,
    ):
        super().__init__()
        self.dimension_reduction = nn.Linear(vision_feature_dim, time_feature_dim)
        self.combine_head = nn.Linear(
            time_feature_num + vision_feature_num, time_feature_num
        )

    def forward(self, F_time: torch.Tensor, F_vision: torch.Tensor) -> torch.Tensor:
        F_vision = self.dimension_reduction(F_vision)
        F_vision = einops.rearrange(F_vision, "b n d -> b 1 d n")
        F_vision = F_vision.expand(-1, F_time.shape[1], -1, -1)

        F_fusion = torch.cat([F_time, F_vision], dim=-1)
        F_fusion = self.combine_head(F_fusion)
        return F_fusion


class CrossModalMultiHeadAttentionCombine(nn.Module):
    def __init__(
        self,
        time_feature_num: int = 42,
        time_feature_dim: int = 256,
        vision_feature_num: int = 197,
        vision_feature_dim: int = 768,
        num_heads: int = 2,
        gate: bool = False,
    ):
        super().__init__()

        model_dim = time_feature_dim

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads  # d_k = d_model/h

        self.Q_construct = nn.Linear(time_feature_dim, model_dim)
        self.K_construct = nn.Linear(vision_feature_dim, model_dim)
        self.V_construct = nn.Linear(vision_feature_dim, model_dim)

        self.W_Q = nn.Linear(model_dim, model_dim)
        self.W_K = nn.Linear(model_dim, model_dim)
        self.W_V = nn.Linear(model_dim, model_dim)
        self.W_O = nn.Linear(model_dim, model_dim)

        self.layer_norm = nn.LayerNorm(model_dim)

        self.gate = gate
        if self.gate:
            self.gate_W = nn.Linear(model_dim, model_dim)
            self.gate_b = nn.Parameter(torch.zeros(model_dim))

    def forward(self, F_time: torch.Tensor, F_vision: torch.Tensor) -> torch.Tensor:
        F_time = einops.rearrange(F_time, "b v d n -> v b d n")
        x_outputs = []

        for var in F_time:
            var = einops.rearrange(var, "b d n -> b n d")
            batch_size = var.size(0)

            Q_origin = self.Q_construct(var)
            K_origin = self.K_construct(F_vision)
            V_origin = self.V_construct(F_vision)

            if self.gate:
                G = torch.sigmoid(self.gate_W(var) + self.gate_b)

            Q = self.W_Q(Q_origin)
            K = self.W_K(K_origin)
            V = self.W_V(V_origin)

            Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
                torch.tensor(self.head_dim, dtype=torch.float32)
            )

            attn_weights = F.softmax(scores, dim=-1)

            attn_output = torch.matmul(attn_weights, V)

            attn_output = (
                attn_output.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.model_dim)
            )
            attn_output = self.W_O(attn_output)
            attn_output = self.layer_norm(Q_origin + attn_output)

            if self.gate:
                attn_output = G * attn_output + (1 - G) * var

            x_outputs.append(attn_output)

        F_fusion = einops.rearrange(torch.stack(x_outputs, dim=1), "v b n d -> b v d n")

        return F_fusion
