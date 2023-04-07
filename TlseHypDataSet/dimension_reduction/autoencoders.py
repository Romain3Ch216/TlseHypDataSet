import torch
import torch.nn as nn
import torch.nn.functional as F
from TlseHypDataSet.utils.spectral import get_continuous_bands, SpectralWrapper


class Encoding:
    def __init__(self, autoencoder, lr, epochs):
        self.autoencoder = autoencoder
        self.optim = torch.optim.Adam(lr=lr)
        self.epochs = epochs

    def fit(self, data_loader):
        for epoch in range(self.epochs):
            for x, _ in data_loader:
                z, r = self.autoencoder(x)
                loss = F.mse_loss(x, r)
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

    def transform(self, data_loader):
        proj = []
        for x, _ in data_loader:
            with torch.no_grad():
                z = self.autoencoder.encode(x)
                proj.append(z)
        proj = torch.cat(proj)
        return proj


class AutoEncoder(nn.Module):
    def __init__(self, x_dim, bbl, int_channels, h_dim, z_dim, dropout):
        super(AutoEncoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.dropout = dropout

        n_bands = get_continuous_bands(bbl)
        convs = {}
        for i in range(len(n_bands)):
            convs[f'conv-{i}'] = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=int_channels, kernel_size=n_bands[i]//5),
                nn.MaxPool1d(kernel_size=3),
                nn.ReLU(),
                nn.Conv1d(in_channels=int_channels, out_channels=int_channels, kernel_size=n_bands[i]//5),
                nn.MaxPool1d(kernel_size=2),
                nn.ReLU()
            )
        self.convs = SpectralWrapper(convs)

        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.out_dim_convs(), h_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(h_dim, z_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim),
            nn.Sigmoid()
        )

    def out_dim_convs(self):
        x = torch.ones((1, 1, self.x_dim))
        x = self.convs(x)
        return x.numel()

    def encode(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.encoder(x)
        return x

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z = self.encode(x)
        x = self.decode(z)
        return z, x