import torch
from torch import nn

from cnn_transformer_model.transformer_model import PositionalEncoding


class CNNTransformer(nn.Module):
    """
    Transformer based decoder model for brain-to-text, using pretrained EENet and Transformer models.

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
        eenet_model,
        transformer_model,
        rnn_dropout=0.0,
        input_dropout=0.0,
        n_layers=4,
        patch_size=0,
        patch_stride=0,
    ):
        super().__init__()

        # Assign pretrained models
        self.eenet_model = eenet_model
        self.transformer_model = transformer_model

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

        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.n_units)

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

    def forward(self, x, day_idx):
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

        # Go through EEGNet and project into transformer input space
        x = self.eenet_model(x)

        # positional encoding
        x = self.pos_encoding(x)
        x = self.transformer_model(x)
        # Prediction head
        logits = self.out(x)

        return logits
