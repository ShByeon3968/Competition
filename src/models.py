import torch
import torch.nn as nn

class LSTM_AE(nn.Module):
    def __init__(self,HIDDEN_DIM_LSTM,NUM_LAYERS,DROPOUT):
        super(LSTM_AE, self).__init__()

        # LSTM feature extractor
        self.lstm_feature = nn.LSTM(
            input_size=1, 
            hidden_size=HIDDEN_DIM_LSTM,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )

        # Encoder modules
        self.encoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM_LSTM, HIDDEN_DIM_LSTM//4),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_LSTM//4, HIDDEN_DIM_LSTM//8),
            nn.ReLU(),
        )

        # Decoder modules
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_DIM_LSTM//8, HIDDEN_DIM_LSTM//4),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_LSTM//4, HIDDEN_DIM_LSTM),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm_feature(x)
        last_hidden = hidden[-1]  # (batch, hidden_dim)

        # AE
        latent_z = self.encoder(last_hidden)
        reconstructed_hidden = self.decoder(latent_z)

        return last_hidden, reconstructed_hidden

class TransformerAE(nn.Module):
    def __init__(self, HIDDEN_DIM_TRANSFORMER:int, NUM_HEADS:int,DROPOUT:int,NUM_LAYERS:int,):
        super(TransformerAE, self).__init__()

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM_TRANSFORMER,
            nhead=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=NUM_LAYERS)

        # Latent representation
        self.latent_fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM_TRANSFORMER, HIDDEN_DIM_TRANSFORMER//4),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_TRANSFORMER//4, HIDDEN_DIM_TRANSFORMER//8),
        )

        # Transformer Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=HIDDEN_DIM_TRANSFORMER,
            nhead=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=NUM_LAYERS)

        # Output reconstruction
        self.reconstruction_fc = nn.Linear(HIDDEN_DIM_TRANSFORMER//8, HIDDEN_DIM_TRANSFORMER)

    def forward(self, x):
        # Encode
        encoded_seq = self.encoder(x)  # (batch, seq_len, hidden_dim)
        latent_z = self.latent_fc(encoded_seq[:, -1, :])  # Use last token for latent representation

        # Decode
        latent_seq = latent_z.unsqueeze(1).repeat(1, x.size(1), 1)  # Repeat latent vector for each time step
        decoded_seq = self.decoder(latent_seq, encoded_seq)  # Transformer Decoder

        reconstructed = self.reconstruction_fc(decoded_seq)
        return latent_z, reconstructed
