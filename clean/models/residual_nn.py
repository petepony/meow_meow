import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.ln(x)
        out = self.fc(out)
        out = self.gelu(out)
        out = self.dropout(out)
        out = out + residual
        return out

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_blocks, emb_dim, emb_num=127, dropout=0.05):
        super().__init__()

        self.emb = nn.Embedding(emb_num, emb_dim)

        self.init_layer = nn.Linear(in_dim + emb_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.soft_plus = nn.Softplus()

    def forward(self, x, stock_id):
        emb = self.emb(stock_id)
        out = torch.cat([x, emb], dim=-1)
        out = self.init_layer(out)

        for block in self.blocks:
            out = block(out)

        out = self.out_layer(out)
        return self.soft_plus(out).flatten()

    @staticmethod
    def criterion(y_pred, y_true):
        return torch.sqrt(torch.mean(torch.square((y_pred - y_true) / y_true)))
