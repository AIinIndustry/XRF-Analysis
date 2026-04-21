import torch
import torch.nn as torch_nn

class MLPClassifier(torch_nn.Module):
    def __init__(self, input_dim=600, num_classes=41):
        super().__init__()
        
        self.network = torch_nn.Sequential(
            torch_nn.Linear(input_dim, 512),
            torch_nn.ReLU(),
            torch_nn.Linear(512, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, 128),
            torch_nn.ReLU(),
            torch_nn.Dropout(0.2), 
            torch_nn.Linear(128, num_classes),
            torch_nn.Sigmoid()    
        )

    def forward(self, x):
        # x shape: (batch, 600)
        return self.network(x)

class CNNClassifier(torch_nn.Module):
    def __init__(self, input_dim=600, num_classes=41):
        super().__init__()
        #Encoder
        self.encoder = torch_nn.Sequential(
            torch_nn.Conv1d(1, 16, kernel_size=3, padding=1),
            torch_nn.ReLU(),
            torch_nn.MaxPool1d(2), # 300
            torch_nn.Conv1d(16, 32, kernel_size=3, padding=1),
            torch_nn.ReLU(),
            torch_nn.MaxPool1d(2)  # 150
        )
        #Decoder
        self.classifier = torch_nn.Sequential(
            torch_nn.Flatten(),
            torch_nn.Linear(32 * 150, 128),
            torch_nn.ReLU(),
            torch_nn.Dropout(0.2), 
            torch_nn.Linear(128, num_classes),
            torch_nn.Sigmoid()     
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.encoder(x)
        return self.classifier(x)

class TransformerClassifier(torch_nn.Module):
    def __init__(self, input_dim=600, num_classes=41, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        self.input_projection = torch_nn.Linear(1, d_model)
        
        encoder_layer = torch_nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = torch_nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = torch_nn.Sequential(
            torch_nn.Flatten(),
            torch_nn.Linear(d_model * input_dim, 128),
            torch_nn.ReLU(),
            torch_nn.Dropout(0.2),
            torch_nn.Linear(128, num_classes),
            torch_nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        x = self.input_projection(x)
        
        x = self.transformer_encoder(x)
        
        return self.classifier(x)