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
            # Removed Sigmoid to support all scaling strategies (e.g. StandardScaler)
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
            
        return logits.squeeze(1)


class TransformerAutoencoder(torch_nn.Module):
    """
    1D Transformer Autoencoder.
    Treats each of the features as a token in a sequence.
    """
    def __init__(self, input_dim: int = 600, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        # Map 1-dim feature to d_model
        self.embedding = torch_nn.Linear(1, d_model)
        
        encoder_layer = torch_nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = torch_nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Map back to 1-dim
        self.decoder = torch_nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: (batch_size, input_dim) -> (batch_size, input_dim, 1)
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x.squeeze(-1)


class ConformerAutoencoder(torch_nn.Module):
    """
    Downsamples with Conv1D, applies Transformer, upsamples with ConvTranspose1d.
    """
    def __init__(self, input_dim: int = 600, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # Downsample: 600 -> 150
        self.conv_encoder = torch_nn.Sequential(
            torch_nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), # 300
            torch_nn.ReLU(),
            torch_nn.Conv1d(16, d_model, kernel_size=5, stride=2, padding=2), # 150
            torch_nn.ReLU()
        )
        
        encoder_layer = torch_nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = torch_nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Upsample: 150 -> 600
        self.conv_decoder = torch_nn.Sequential(
            torch_nn.ConvTranspose1d(d_model, 16, kernel_size=5, stride=2, padding=2, output_padding=1), # 300
            torch_nn.ReLU(),
            torch_nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1), # 600
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # conv_encoder output: (batch_size, d_model, 150)
        encoded = self.conv_encoder(x)
        
        # Transformer expects (batch_size, seq_len, d_model) if batch_first=True
        encoded = encoded.permute(0, 2, 1)
        transformed = self.transformer(encoded)
        
        # Back to (batch_size, d_model, seq_len)
        transformed = transformed.permute(0, 2, 1)
        
        decoded = self.conv_decoder(transformed)
        return decoded.squeeze(1)


class ResidualBlock1D(torch_nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = torch_nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = torch_nn.BatchNorm1d(channels)
        self.relu = torch_nn.ReLU(inplace=True)
        self.conv2 = torch_nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch_nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet1DAutoencoder(torch_nn.Module):
    """
    Autoencoder using 1D Residual Blocks.
    """
    def __init__(self, input_dim: int = 600):
        super().__init__()
        
        # Encoder
        self.encoder_initial = torch_nn.Sequential(
            torch_nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2, bias=False), # 300
            torch_nn.BatchNorm1d(32),
            torch_nn.ReLU(inplace=True)
        )
        self.encoder_res1 = ResidualBlock1D(32)
        
        self.encoder_down = torch_nn.Sequential(
            torch_nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False), # 150
            torch_nn.BatchNorm1d(64),
            torch_nn.ReLU(inplace=True)
        )
        self.encoder_res2 = ResidualBlock1D(64)
        
        # Decoder
        self.decoder_res1 = ResidualBlock1D(64)
        self.decoder_up1 = torch_nn.Sequential(
            torch_nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False), # 300
            torch_nn.BatchNorm1d(32),
            torch_nn.ReLU(inplace=True)
        )
        
        self.decoder_res2 = ResidualBlock1D(32)
        self.decoder_final = torch_nn.ConvTranspose1d(32, 1, kernel_size=5, stride=2, padding=2, output_padding=1) # 600

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.encoder_initial(x)
        x = self.encoder_res1(x)
        
        x = self.encoder_down(x)
        x = self.encoder_res2(x)
        
        x = self.decoder_res1(x)
        x = self.decoder_up1(x)
        
        x = self.decoder_res2(x)
        x = self.decoder_final(x)
        
        return x.squeeze(1)


class LSTMAutoencoder(torch_nn.Module):
    """
    RNN (LSTM) Autoencoder. Treats features as a sequence.
    """
    def __init__(self, input_dim: int = 600, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Encoder: returns sequence of hidden states
        self.encoder = torch_nn.LSTM(input_size=1, hidden_size=hidden_dim, 
                                     num_layers=num_layers, batch_first=True)
        
        # Decoder: processes the hidden states to reconstruct
        self.decoder = torch_nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                                     num_layers=num_layers, batch_first=True)
        
        self.output_layer = torch_nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch, input_dim) -> (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        
        # Encode
        encoded, (hn, cn) = self.encoder(x)
        
        # Decode (can just pass encoded sequence to decoder)
        decoded, _ = self.decoder(encoded)
        
        # Map back to 1-dim
        out = self.output_layer(decoded)
        return out.squeeze(-1)


class LinearAutoencoder(torch_nn.Module):
    """
    A simple Linear/PCA-like autoencoder.
    """
    def __init__(self, input_dim: int = 600, hidden_dim: int = 64):
        super().__init__()
        self.encoder = torch_nn.Linear(input_dim, hidden_dim)
        self.decoder = torch_nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
