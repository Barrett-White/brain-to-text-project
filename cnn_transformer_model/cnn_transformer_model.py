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
        cnn_model,
        transformer_model,
        temporal_bin,
        pretrained_cnn: bool = False,
    ):
        super().__init__()
        self.pretrained_cnn = pretrained_cnn
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_days = n_days
        self.n_classes = n_classes
        self.temporal_bin = temporal_bin

        self.cnn = cnn_model
        self.transformer_model = transformer_model

        d_model = self.transformer_model.config.d_model

        # For eegnet or pretrained cnn
        if not pretrained_cnn:
            self.proj_to_t5_eegnet = torch.nn.Linear(n_units, d_model)
        else:
            self.proj_to_t5_cnn = torch.nn.Linear(1000, d_model)
        self.day_embeddings = torch.nn.Embedding(n_days, d_model)

        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.rnn_dropout = torch.nn.Dropout(rnn_dropout)

        self.out = torch.nn.Linear(d_model, n_classes)

    def forward(self, features, day_indices):
        bin_len = self.temporal_bin
        if not self.pretrained_cnn:
            B, T, C = features.shape

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

        else:
            x = features
            B, C, T, _ = features.shape
            S = (T + bin_len - 1) // bin_len

        eeg_feat = self.cnn(x)

        if not self.pretrained_cnn:
            eeg_feat = eeg_feat.view(B, S, self.n_units)

            h = self.proj_to_t5_eegnet(eeg_feat)
        else:
            h = self.proj_to_t5_cnn(eeg_feat)

        day_emb = self.day_embeddings(day_indices).unsqueeze(1)
        h = h + day_emb

        h = self.input_dropout(h)

        if not self.pretrained_cnn:
            attn_mask = torch.ones(B, S, dtype=torch.long, device=h.device)
            encoder_outputs = self.transformer_model.encoder(
                inputs_embeds=h,
                attention_mask=attn_mask,
            )
        else:
            # We don't need an attention mask because we didn't do padding in this version
            encoder_outputs = self.transformer_model.encoder(
                inputs_embeds=h,
            )
        hidden = encoder_outputs.last_hidden_state
        hidden = self.rnn_dropout(hidden)

        logits = self.out(hidden)
        return logits
