import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, seq_len=30):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim, # Input to decoder is the latent vector repeated
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        
        # Encode
        _, (hidden, _) = self.encoder(x)
        # hidden shape: (num_layers, batch, hidden_dim)
        # We take the last layer's hidden state as the latent representation
        latent = hidden[-1] # (batch, hidden_dim)
        
        # Repeat latent vector for each time step in sequence
        latent_repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq_len, hidden_dim)
        
        # Decode
        decoded_lstm, _ = self.decoder(latent_repeated)
        # decoded_lstm shape: (batch, seq_len, hidden_dim)
        
        # Map back to input space
        reconstructed = self.output_layer(decoded_lstm)
        return reconstructed

class LSTMAutoencoderDetector:
    def __init__(self, seq_len: int = 30, hidden_dim: int = 32, epochs: int = 20, batch_size: int = 64):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        
    def create_sequences(self, data, seq_len):
        xs = []
        for i in range(len(data) - seq_len):
            x = data[i:(i + seq_len)]
            xs.append(x)
        return np.array(xs)
        
    def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        X_seq = self.create_sequences(X_scaled, self.seq_len)
        X_tensor = torch.FloatTensor(X_seq)
        
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_dim = X.shape[1]
        self.model = LSTMAutoencoder(input_dim, self.hidden_dim, seq_len=self.seq_len)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Train
        self.model.train()
        for epoch in range(self.epochs):
            for batch_in, batch_target in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_in)
                loss = criterion(output, batch_target)
                loss.backward()
                optimizer.step()
                
        # Predict
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            # MSE per sequence
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=(1, 2)).numpy()
            
        # Pad the beginning with NaNs or zeros to match original index length
        # The first seq_len rows don't have a full sequence ending at them (depending on how we define it)
        # Here, the i-th sequence corresponds to data[i : i+seq_len]. 
        # Let's align the score with the END of the sequence.
        pad_width = self.seq_len
        mse_padded = np.pad(mse, (pad_width, 0), mode='constant', constant_values=np.nan)
        mse_padded = mse_padded[:len(df)] # Trim if needed, though padding should match
        
        # Threshold
        valid_mse = mse_padded[~np.isnan(mse_padded)]
        threshold = np.percentile(valid_mse, 99)
        is_anomaly = mse_padded > threshold
        
        return pd.DataFrame({'score': mse_padded, 'is_anomaly': is_anomaly}, index=df.index)
