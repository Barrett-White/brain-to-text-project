import torch


class CNNTransformer(torch.nn.Module):
    def __init__(
        self,
        neural_dim,
        n_units,
        n_days,
        n_classes,
        rnn_dropout,
        input_dropout,
        eenet_model,
        transformer_model,
        temporal_bin,
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.temporal_bin = temporal_bin

        self.eegnet = eenet_model
        self.transformer_model = transformer_model

        d_model = self.transformer_model.config.d_model

        self.proj_to_t5 = torch.nn.Linear(n_units, d_model)
        self.day_embeddings = torch.nn.Embedding(n_days, d_model)

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.rnn_dropout = torch.nn.Dropout(rnn_dropout)

        self.out = torch.nn.Linear(d_model, n_classes)

    def forward(self, features, day_indices):
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
