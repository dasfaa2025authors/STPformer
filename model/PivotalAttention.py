import torch.nn as nn
import torch
# from torchinfo import summary
from lib.masking import ProbMask
import numpy as np
from math import sqrt
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class AttentionLayer_Hawkes(nn.Module):
    """
    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)


        # step 1 get delta_T based on the time seg with tgt_length
        delta_T = torch.linspace(0, tgt_length-1, tgt_length).cuda(0)

        # step 2 compute decay signal with decay_rate
        decay_rate = F.normalize(torch.exp((1.0 * delta_T)).cuda(0),p=2, dim=0)

        attn_hawkes = torch.mul(attn_score, decay_rate)

        attn_score = torch.add(attn_score,attn_hawkes)


        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)

        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)


        out = self.out_proj(out)

        return out

class SelfAttentionLayer_Hawkes(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        # self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.attn = AttentionLayer_Hawkes(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )

        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class PivotalAttention(nn.Module):
    """
    """
    def __init__(self, mask_flag=True, sc_factor=10, attention_dropout=0.1, output_attention=False,scale=None):
        super(PivotalAttention, self).__init__()
        D = 152
        self.fc_query = torch.nn.Linear(D, D)
        self.fc_key = torch.nn.Linear(D, D)
        self.fc_value = torch.nn.Linear(D, D)

        self.factor = sc_factor # spatial sampling factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def _prob_QK(self, Q, K, sample_k, n_top):  # Top_c: sc*ln(N)

        # Q, K, V (B,T,N,D)
        B, H, L_K, E = K.shape  # (B,T,N,D)
        _, _, L_Q, _ = Q.shape  # (B,T,N,D)

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # (B, T, N, N, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # (N, Nu) Nu = sc * ln(N)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # (B, T, N, Nu, D)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()  # (B, T, Nu, D)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # (B, T, N)
        M_top = M.topk(n_top, sorted=False)[1]  # (B,T,Nu)
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # (B, T, Nu, D)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape  # (B, T, N, D)
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)  # (N cumsum)
        return contex  # (B, T, N, D)

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape  # (B, T, N, D)
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # (B, T, N, D)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, X, attn_mask):

        queries = F.relu(self.fc_query(X))  # FC -> (B,T,N,D)
        keys = F.relu(self.fc_key(X))  # FC-> (B,T,N,D)
        values = F.relu(self.fc_value(X))  # FC -> (B,T,N,D)

        B, T, N_Q, D = queries.shape  # (B, T, N, D)
        _, _, N_K, _ = keys.shape

        NU_part = self.factor * np.ceil(np.log(N_K)).astype('int').item()  # sc*ln(N)
        Nu = self.factor * np.ceil(np.log(N_Q)).astype('int').item()  # sc*ln(N)
        NU_part = NU_part if NU_part < N_K else N_K
        Nu = Nu if Nu < N_Q else N_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=NU_part, n_top=Nu)  # (B, T, Nu, N) (B, T, N)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # (B, T, Nu, N)

        # get the context
        context = self._get_initial_context(values, N_Q)  # (B, T, N, D) cumsum in dim N

        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, N_Q, attn_mask)

        return context.transpose(2, 1).contiguous(), attn


class Pivotalattentionlayer(nn.Module):
    def __init__(self,n_heads, mix=True):
        super(Pivotalattentionlayer, self).__init__()
        D = 152 # model dim for PEMS08
        self.n_heads = n_heads
        self.inner_attention = PivotalAttention(mask_flag=True, sc_factor=5,
                                                scale=None, attention_dropout=0.1, output_attention=False)

        self.feed_forward = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, D),
        )

        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.mix = mix

    def forward(self, x, attn_mask):

        residual = x
        out, attn = self.inner_attention(
           x, attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        # out = out.view(B, L, -1)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (B, ..., T, D)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        return out, attn

