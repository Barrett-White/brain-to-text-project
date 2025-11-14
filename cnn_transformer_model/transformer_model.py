import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Standard sine-cosine positional encoding for sequences.
    Produces [B, T, d_model] same shape as input, added elementwise.
    """

    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        """
        x: [B, T, d_model]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerDecoder(nn.Module):
    """
    Transformer version of the Brain-to-Text decoder.

    It mirrors GRUDecoder:
    - day-specific linear layers
    - optional patching over time
    - CTC-compatible logits: [B, T', n_classes]
    - forward(x, day_idx, states=None, return_state=False)
    """

    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=4,
        patch_size=0,
        patch_stride=0,
        n_heads=8,
        dim_feedforward=None,
    ):
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days

        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.day_layer_activation = nn.Softsign()

        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        # Input size before projection
        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        # Project to transformer
        self.input_proj = nn.Linear(self.input_size, self.n_units)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.n_units)

        # Transformer Encoder
        if dim_feedforward is None:
            dim_feedforward = 4 * self.n_units

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.n_units,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=self.rnn_dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def _apply_day_layers(self, x, day_idx):
        """
        x: [B, T, D]
        day_idx: [B] (int day index per trial)
        """
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(
            1
        )

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        return x

    def _apply_patching(self, x):
        """
        Optional patching over time, copied from GRUDecoder logic.

        x: [B, T, D]
        returns: [B, T_patch, D * patch_size] if patching enabled,
                 else [B, T, D]
        """
        if self.patch_size <= 0:
            return x

        x = x.unsqueeze(1)
        x = x.permute(0, 3, 1, 2)
        x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
        x_unfold = x_unfold.squeeze(2)
        x_unfold = x_unfold.permute(0, 2, 3, 1)
        x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)
        return x

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        x: [B, T, neural_dim]
        day_idx: [B] int indices of day

        returns:
          logits: [B, T', n_classes]
          (optionally) None as "hidden state", to match GRUDecoder interface
        """
        # Day specific normalization
        x = self._apply_day_layers(x, day_idx)

        x = self._apply_patching(x)

        # Project to transformer d_model
        x = self.input_proj(x)

        # positional encoding
        x = self.pos_encoding(x)
        x = self.transformer(x)
        # Prediction head
        logits = self.out(x)

        if return_state:
            return logits, None
        return logits
