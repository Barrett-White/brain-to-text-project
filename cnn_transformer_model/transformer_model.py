import torch
from braindecode.models import EEGNetv4
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Tokenizer

# Currently using code from https://towardsdatascience.com/convolutional-neural-networks-for-eeg-brain-computer-interfaces-9ee9f3dd2b81/


# We can utilize the actual EEGNet here: https://huggingface.co/PierreGtch/EEGNetv4


class CNNDecoder(nn.Module):
    """
    Defines the CNN & Transformer encoder/ decoder

    This class combines day-specific input layers, a GRU, and an output classification layer
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
        n_layers=5,
        patch_size=0,
        patch_stride=0,
        hidden_size=500,
    ):
        """
        neural_dim  (int)      - number of channels in a single timestep (e.g. 512)
        n_units     (int)      - number of hidden units in each recurrent layer - equal to the size of the hidden state
        n_days      (int)      - number of days in the dataset
        n_classes   (int)      - number of classes
        rnn_dropout    (float) - percentage of units to droupout during training
        input_dropout (float)  - percentage of input units to dropout during training
        n_layers    (int)      - number of recurrent layers
        patch_size  (int)      - the number of timesteps to concat on initial input layer - a value of 0 will disable this "input concat" step
        patch_stride(int)      - the number of timesteps to stride over when concatenating initial input
        hidden_size (int)      - the hidden size of the transformer model
        eenet_model (nn.Module) - pretrained EEGNet model for feature extraction
        transformer_model (nn.Module) - pretrained Transformer model for sequence modeling
        """
        super(CNNDecoder, self).__init__()

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

        # Parameters for the day-specific input layers
        self.day_layer_activation = nn.Softsign()  # basically a shallower tanh

        # Set weights for day layers to be identity matrices so the model can learn its own day-specific transformations
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )

        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim

        # If we are using "strided inputs", then the input size of the first recurrent layer will actually be in_size * patch_size
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,  # The first dim of our input is the batch dim
            bidirectional=False,
        )

        # Set recurrent units to have orthogonal param init and input layers to have xavier init
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Prediciton head. Weight init to xavier
        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # following is from https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = nn.Parameter(
            torch.randn(1, self.patch_embeddings.num_patches + 1, hidden_size)
        )

        # Now, MLP and then transformer architecture

        # Learnable initial hidden states
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        x        (tensor)  - batch of examples (trials) of shape: (batch_size, time_series_length, neural_dim)
        day_idx  (tensor)  - tensor which is a list of day indexs corresponding to the day of each example in the batch x.
        """
        # Apply day-specific layer to (hopefully) project neural data from the different days to the same latent space
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(
            1
        )

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)

        # Apply dropout to the ouput of the day specific layer
        if self.input_dropout > 0:
            x = self.day_layer_dropout(x)

        # (Optionally) Perform input concat operation
        if self.patch_size > 0:
            x = x.unsqueeze(1)  # [batches, 1, timesteps, feature_dim]
            x = x.permute(0, 3, 1, 2)  # [batches, feature_dim, 1, timesteps]

            # Extract patches using unfold (sliding window)
            x_unfold = x.unfold(
                3, self.patch_size, self.patch_stride
            )  # [batches, feature_dim, 1, num_patches, patch_size]

            # Remove dummy height dimension and rearrange dimensions
            x_unfold = x_unfold.squeeze(
                2
            )  # [batches, feature_dum, num_patches, patch_size]
            x_unfold = x_unfold.permute(
                0, 2, 3, 1
            )  # [batches, num_patches, patch_size, feature_dim]

            # Flatten last two dimensions (patch_size and features)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # Determine initial hidden states
        if states is None:
            states = self.h0.expand(
                self.n_layers, x.shape[0], self.n_units
            ).contiguous()

        # Pass input through CNN
        with torch.no_grad():
            x = self.eenet_model(x)

        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)

        # Pass through transformer model
        with torch.no_grad():
            output = self.transformer_model(x)

        # Compute logits
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits


if __name__ == "__main__":
    # https://neurotechlab.socsci.ru.nl/resources/pretrained_imagery_models/
    path = hf_hub_download(
        repo_id="PierreGtch/EEGNetv4",
        filename="EEGNetv4_Lee2019_MI/model-params.pkl",
    )
    net = EEGNetv4(3, 2, 385).eval()
    net.load_state_dict(torch.load(path, map_location="cpu"))

    # Transformer architecture from https://www.datacamp.com/tutorial/flan-t5-tutorial
    # Load the tokenizer, model, and data collator
    MODEL_NAME = "google/flan-t5-base"

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # The full model forward includes using these models in prediction
    full_model = CNNDecoder(
        neural_dim=385,
        n_units=128,
        n_days=5,
        n_classes=2,
        eenet_model=net,
        transformer_model=model,
    )
