import torch
from torch import nn


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

        d_model = self.transformer_model.config.d_model

        self.proj_to_t5 = torch.nn.Linear(n_units, d_model)
        self.day_embeddings = torch.nn.Embedding(n_days, d_model)

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.rnn_dropout = torch.nn.Dropout(rnn_dropout)

        self.out = torch.nn.Linear(d_model, n_classes)

    def forward(self, features, day_indices):
        """
        features: [B, T, neural_dim]
        day_indices: [B] int indices of day

        returns:
          logits: [B, T', n_classes]
          (optionally) None as "hidden state", to match GRUDecoder interface
        """
        B, T, C = features.shape
        bin_len = self.temporal_bin

        S = (T + bin_len - 1) // bin_len
        T_eff = S * bin_len
        if T_eff > T:
            pad = torch.zeros(
                B, T_eff - T, C, device=features.device, dtype=features.dtype
            )
            x = torch.cat([features, pad], dim=1)
        else:
            x = features

        x = x.view(B, S, bin_len, C).permute(0, 1, 3, 2)
        x = x.reshape(B * S, C, bin_len)

        eeg_feat = self.eegnet(x)
        eeg_feat = eeg_feat.view(B, S, self.n_units)

        h = self.proj_to_t5(eeg_feat)

        day_emb = self.day_embeddings(day_indices).unsqueeze(1)
        h = h + day_emb

        h = self.input_dropout(h)

        attn_mask = torch.ones(B, S, dtype=torch.long, device=h.device)
        encoder_outputs = self.transformer_model.encoder(
            inputs_embeds=h,
            attention_mask=attn_mask,
        )
        hidden = encoder_outputs.last_hidden_state
        hidden = self.rnn_dropout(hidden)

        logits = self.out(hidden)
        return logits
