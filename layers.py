import torch
import torch.nn as nn
import torch.nn.functional as F


class BertPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x):
        first_token_tensor = x[:, 0]
        pooled_output = self.fc(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertMLMHead(nn.Module):
    def __init__(self, d_model, vocab_size, norm_eps=1e-12):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model, eps=norm_eps)
        )
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        output = self.decoder(self.transform((hidden_states)))
        return output


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, pd=0.1, norm_eps=1e-12):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=norm_eps)
        self.dropout = nn.Dropout(pd)

    def forward(self, input_ids, token_type_ids=None):
        seq_len = input_ids.size(1)
        batch_size = input_ids.size(0)

        position_ids = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        x = self.word_embedding(input_ids)
        x += self.positional_embedding(position_ids)
        x += self.token_type_embedding(token_type_ids)
        x = self.layer_norm(x)
        return self.dropout(x)
