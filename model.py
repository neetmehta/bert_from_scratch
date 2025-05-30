import torch
import torch.nn as nn
import torch.nn.functional as F


class Bert(nn.Module):

    def __init__(
        self,
        num_blocks,
        num_heads,
        d_model,
        vocab_size,
        d_ff,
        max_len=512,
        pd=0.1,
        norm_eps=1e-12,
    ) -> None:
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.sentence_embedding = nn.Embedding(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model, norm_eps)
        self.dropout = nn.Dropout(pd)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_ff,
                    dropout=pd,
                    activation=F.gelu,
                    layer_norm_eps=norm_eps,
                    batch_first=True,
                    bias=True,
                )
                for _ in range(num_blocks)
            ]
        )
        self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.word_embedding(x)
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x += self.positional_embedding(pos)
        x += self.sentence_embedding(
            torch.zeros_like(x[:, :, 0], dtype=torch.long, device=x.device)
        )
        x = self.layer_norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return x
