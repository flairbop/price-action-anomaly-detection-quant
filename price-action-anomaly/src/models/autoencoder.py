import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=16):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderDetector:
    def __init__(self, encoding_dim: int = 16, epochs: int = 50, batch_size: int = 64):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = None
        
    def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_dim = X.shape[1]
        self.model = Autoencoder(input_dim, self.encoding_dim)
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
                
        # Predict (Reconstruction Error)
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1).numpy()
            
        # Thresholding (e.g., 99th percentile)
        threshold = np.percentile(mse, 99)
        is_anomaly = mse > threshold
        
        return pd.DataFrame({'score': mse, 'is_anomaly': is_anomaly}, index=df.index)
