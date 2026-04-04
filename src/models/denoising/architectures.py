import torch
import torch.nn as torch_nn
import torch.nn.functional as F

class MLPAutoencoder(torch_nn.Module):
    """
    A simple Fully Connected Autoencoder.
    """
    def __init__(self, input_dim: int = 600):
        super().__init__()
        # Encoder
        self.encoder = torch_nn.Sequential(
            torch_nn.Linear(input_dim, 512),
            torch_nn.ReLU(),
            torch_nn.Linear(512, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, 128),
            torch_nn.ReLU()
        )
        # Decoder
        self.decoder = torch_nn.Sequential(
            torch_nn.Linear(128, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, 512),
            torch_nn.ReLU(),
            torch_nn.Linear(512, input_dim),
            torch_nn.Sigmoid()  # Assuming data is scaled to [0, 1]
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CNNAutoencoder(torch_nn.Module):
    """
    A standard 1D CNN Autoencoder.
    """
    def __init__(self, input_dim: int = 600):
        super().__init__()
        # Encoder
        self.encoder = torch_nn.Sequential(
            torch_nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), # output: 300
            torch_nn.ReLU(),
            torch_nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # output: 150
            torch_nn.ReLU(),
            torch_nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2), # output: 75
            torch_nn.ReLU()
        )
        # Decoder
        self.decoder = torch_nn.Sequential(
            torch_nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1), # output: 150
            torch_nn.ReLU(),
            torch_nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # output: 300
            torch_nn.ReLU(),
            torch_nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),  # output: 600
            torch_nn.Sigmoid() # Assuming data is scaled to [0, 1]
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)


class DoubleConv(torch_nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = torch_nn.Sequential(
            torch_nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(out_channels),
            torch_nn.ReLU(inplace=True),
            torch_nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch_nn.BatchNorm1d(out_channels),
            torch_nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet1D(torch_nn.Module):
    """
    A 1D U-Net architecture, excellent for capturing local peaks and global context.
    """
    def __init__(self, input_dim: int = 600, n_channels: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.input_dim = input_dim

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = torch_nn.Sequential(torch_nn.MaxPool1d(2), DoubleConv(16, 32))
        self.down2 = torch_nn.Sequential(torch_nn.MaxPool1d(2), DoubleConv(32, 64))
        self.down3 = torch_nn.Sequential(torch_nn.MaxPool1d(2), DoubleConv(64, 128))
        
        self.up1 = torch_nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)
        
        self.up2 = torch_nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(64, 32)
        
        self.up3 = torch_nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(32, 16)
        
        self.outc = torch_nn.Conv1d(16, 1, kernel_size=1)
        self.sigmoid = torch_nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # padding logic to ensure the dimension is divisible by 8 for 3 poolings
        orig_dim = x.shape[2]
        pad_size = (8 - orig_dim % 8) % 8
        if pad_size > 0:
            x = F.pad(x, (0, pad_size))

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        logits = self.outc(x)
        
        if pad_size > 0:
            logits = logits[:, :, :-pad_size]
            
        return self.sigmoid(logits).squeeze(1)
