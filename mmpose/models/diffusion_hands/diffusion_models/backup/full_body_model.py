from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F

from .rotary_embedding_torch import RotaryEmbedding
from .utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, nframes, n_features = x.shape
        assert nframes == 196
        x = x.permute((1, 0, 2)) # [seq_len, bs, n_features]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x



class OutputProcess(nn.Module):
    def __init__(self, data_rep, output_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = output_feats
        self.latent_dim = latent_dim
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.poseFinal(output)  # [seqlen, bs, 150]
        output = output.reshape(nframes, bs, self.input_feats)
        output = output.permute(1, 0, 2)  # [bs, n_frames, nfeats]
        return output

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class MDMDenoiserNetwork(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 196,  # 5 seconds, 30 fps
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 512,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        arch='trans_enc',
        **kwargs
    ) -> None:

        super(MDMDenoiserNetwork, self).__init__()

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding processing
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim),)

        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )
        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        
        self.final_layer = nn.Linear(latent_dim, output_feats)

    def guided_forward(self, x, cond_embed, times, guidance_weight):
        unc = self.forward(x, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_embed, times, cond_drop_prob=0)

        return unc + (conditioned - unc) * guidance_weight

    def forward(
        self, x: Tensor, cond_embed: Tensor, times: Tensor, cond_drop_prob: float = 0.0
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)
        # add the positional embeddings of the input sequence to provide temporal information
        x = self.abs_pos_encoding(x)

        # create music conditional embedding with conditional dropout
        keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_projection(cond_embed)
        # encode tokens
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        # should not use any pooling here!!
        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)  # pool in terms of sequence length, not good!
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # create the diffusion timestep embedding, add the extra music projection
        t_hidden = self.time_mlp(times)

        # project to attention and FiLM conditioning
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        # FiLM conditioning
        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        # cross-attention conditioning
        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # Pass through the transformer decoder
        # attending to the conditional embedding
        output = self.seqTransDecoder(x, cond_tokens, t)

        output = self.final_layer(output)
        return output
