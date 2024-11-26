import torch.nn as nn
import torch
from torchinfo import summary
import torch.nn.functional as F
from PivotalAttention import Pivotalattentionlayer, SelfAttentionLayer_Hawkes, SelfAttentionLayer
from lib.utils import HuberLoss

class STPformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)


        # Pivotal Temporal Attention Module (PTA)
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer_Hawkes(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(3)
            ]
        )

        # Pivotal Spatial Attention Module (PSA)
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(2)
            ]
        )

        
        self.attn_layers_s_pivotal = nn.ModuleList(
            [
                Pivotalattentionlayer(n_heads=num_heads) for _ in range(1)
            ]
        )
        

    def forward(self, x):
        # x: (B, T, N, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (B, T, N, d_a)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (B, T, N, d_t)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (B, T, N, d_t)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            ) # (B, T, N, d_a)
            features.append(adp_emb)

        # D = d_t + d_s + d_a
        x = torch.cat(features, dim=-1)  # (B, T, N, D)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)

        residu = x

        for attn in self.attn_layers_s:
            x = attn(x, dim=2)

        for attn in self.attn_layers_s_pivotal:
            x_pivotal, spatial_attn = attn(residu,None)

        x = torch.add(x, x_pivotal)

        # (batch_size, in_steps, num_nodes, model_dim)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (B, N, T, D)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (B, H, N, 1)
        else:
            out = x.transpose(1, 3)  # (B, D, N, T)
            out = self.temporal_proj(
                out
            )  # (B, D, N, 1)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (B, H, N, 1)

        return out
