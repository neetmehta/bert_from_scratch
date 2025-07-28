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


class BertEncoder(nn.Module):

    def __init__(
        self,
        num_blocks,
        d_model,
        num_heads,
        d_ff,
        p_d,
        activation,
        layer_norm_eps,
        batch_first,
        bias,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_ff,
                    dropout=p_d,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    batch_first=batch_first,
                    bias=bias,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x, attention_mask=None, token_type_ids=None):
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        key_padding_mask = key_padding_mask.to(x.device) if key_padding_mask is not None else None
        for block in self.blocks:
            x = block(x, src_key_padding_mask=key_padding_mask)
        return x


class BertPretrainingHeads(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.mlm_head = BertMLMHead(d_model, vocab_size)
        self.nsp_head = nn.Linear(d_model, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.mlm_head(sequence_output)
        seq_relationship_score = self.nsp_head(pooled_output)
        return prediction_scores, seq_relationship_score


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
        self.bert_encoder = BertEncoder(
            num_blocks, d_model, num_heads, d_ff, pd, F.gelu, norm_eps, True, True
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
        x = self.bert_encoder(
            x, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooler_output = self.pooler(x) if self.pooler is not None else None
        return x, pooler_output


class BertForPretraining(nn.Module):

    def __init__(
        self,
        num_blocks,
        num_heads,
        d_model,
        vocab_size,
        d_ff,
        max_len=512,
        pd=0,
        norm_eps=1e-12,
        initialize=True,
        add_pooler=True,
        device=None,
    ) -> None:
        super().__init__()
        self.bert = Bert(
            num_blocks,
            num_heads,
            d_model,
            vocab_size,
            d_ff,
            max_len,
            pd,
            norm_eps,
            initialize,
            add_pooler,
        )
        self.cls = BertPretrainingHeads(d_model, vocab_size)
        self.cls.mlm_head.decoder.weight = (
            self.bert.bert_embedding.word_embedding.weight
        )
        self.loss = nn.CrossEntropyLoss(
            ignore_index=-100, reduction="mean")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if initialize:
            self.init_parameters()

    def init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch):
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        next_sentence_label = batch["next_sentence_label"].to(self.device)
        sequence_output, pooled_output = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output
        )
        loss_mlm = self.loss(
            prediction_scores.view(-1, prediction_scores.size(-1)),
            labels.view(-1),
        )
        loss_nsp = self.loss(
            seq_relationship_score.view(-1, seq_relationship_score.size(-1)),
            next_sentence_label.view(-1),
        )
        total_loss = loss_mlm + loss_nsp
        return {'loss': total_loss,
                'prediction_scores': prediction_scores,
                'seq_relationship_score': seq_relationship_score}
