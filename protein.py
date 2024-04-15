import torch.nn as nn
import torch
from GNN import LayerNorm
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_head, dropout):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_size % num_head == 0
        self.d_per_head = hidden_size // num_head
        self.num_head = num_head
        self.hidden_size = hidden_size
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose(self, x):
        # batch_size, seq_len, hidden_size -> batch_size, head_num, seq_len, hidden_size_per_head
        return x.view(x.size()[0], x.size()[1], self.num_head, self.d_per_head).permute(0, 2, 1, 3)

    def reverse_transpose(self, x):
        # batch_size, head_num, seq_len, hidden_size_per_head -> batch_size, seq_len, hidden_size
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(x.size()[0], x.size()[1], self.hidden_size)

    def forward(self, hidden_states, mask):
        q = self.transpose(self.w_q(hidden_states))
        k = self.transpose(self.w_k(hidden_states))
        v = self.transpose(self.w_v(hidden_states))
        score = torch.matmul(q, k.transpose(-1, -2))
        score /= math.sqrt(self.d_per_head)
        score += mask
        attn = self.dropout(nn.Softmax(dim=-1)(score))
        c = torch.matmul(attn, v)
        # c = c.permute(0, 2, 1, 3).contiguous()
        c = self.reverse_transpose(c)
        return c


class MLP(torch.nn.Module):
    def __init__(self, in_feats, out_feats, hidden_size, dropout=0.):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size, out_feats),
        )

    def forward(self, x):
        x_out = self.mlp(x)
        return x_out


class ProTrans(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, num_attn_layer, num_lstm_layer, num_attn_heads,
                 ffn_hidden_size, lstm, dropout):
        super(ProTrans, self).__init__()
        self.lstm = lstm
        self.embedding = Embeddings(vocab_size, hidden_size, max_position_size, dropout)
        self.layers = nn.ModuleList(
            [
                Encoder(hidden_size,
                        ffn_hidden_size,
                        num_attn_heads,
                        dropout)
                for _ in range(num_attn_layer)
            ]
        )
        if self.lstm:
            self.lstm_layer = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_lstm_layer,
                                      batch_first=True, dropout=dropout)
        self.mlp = MLP(hidden_size, hidden_size, ffn_hidden_size)

    def forward(self, protein, prot_mask):
        prot_encode = self.embedding(protein)

        prot_mask = prot_mask.unsqueeze(1).unsqueeze(2)
        prot_mask = (1.0 - prot_mask) * -10000.0

        for layer in self.layers:
            output = layer(prot_encode.float(), prot_mask.float())
        if self.lstm:
            self.lstm_layer.flatten_parameters()
            prot_encode, _ = self.lstm_layer(output)
            output = self.mlp(prot_encode)

        return output


class Encoder(nn.Module):
    def __init__(self, embed_dim, ffn_hidden_size, num_attn_heads, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadSelfAttention(hidden_size=embed_dim, num_head=num_attn_heads, dropout=dropout)
        self.layer_norm1 = LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.output = MLP(embed_dim, embed_dim, ffn_hidden_size, dropout)
        self.layer_norm2 = LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, hidden_states, prot_mask):
        attn_output = self.attention(hidden_states, prot_mask)
        output = self.dropout1(attn_output)
        output = self.layer_norm1(output + hidden_states)

        mlp_output = self.output(output)
        mlp_output = self.dropout2(mlp_output)
        layer_output = self.layer_norm2(mlp_output + output)
        return layer_output


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size + 1, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
