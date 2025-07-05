import torch
from torch import nn
from layers import BertEmbedding, BertPooler, BertMLMHead
from torch.nn import functional as F


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
        initialize=True,
        add_pooler=True,
    ) -> None:
        super().__init__()
        self.bert_embedding = BertEmbedding(vocab_size, d_model, max_len, pd, norm_eps)
        self.max_len = max_len
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
        self.pooler = BertPooler(d_model) if add_pooler else None

        if initialize:
            self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, attention_mask=None, token_type_ids=None):
        x = self.bert_embedding(x, token_type_ids=token_type_ids)
        # src_key_padding_mask is True for padded positions
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        for block in self.blocks:
            x = block(x, src_key_padding_mask=key_padding_mask)

        pooler_output = self.pooler(x) if self.pooler is not None else None
        return x, pooler_output


class BertForMLM(nn.Module):

    def __init__(
        self, num_blocks, num_heads, d_model, vocab_size, d_ff, max_len=512, pd=0
    ):
        super().__init__()
        self.bert = Bert(
            num_blocks,
            num_heads,
            d_model,
            vocab_size,
            d_ff,
            max_len,
            pd,
            add_pooler=False,
        )
        self.mlm_head = BertMLMHead(d_model, vocab_size)

    def forward(self, x, attention_mask=None, token_type_ids=None):
        x, _ = self.bert(
            x, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        x = self.mlm_head(x)
        return x
