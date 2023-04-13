from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, k, input_size, emb_size=3):
        super().__init__()

        self.input_size = input_size
        self.k = k
        
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size*k, 128), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, emb_size),   
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_size, 12), 
            nn.ReLU(),
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_size*k),
            nn.Sigmoid(),       
        )

    def forward(self, x):
        # x shape: B * input_size * k
        x = x.reshape(x.shape[0],-1)
        encoded = self.encoder(x) 
        decoded = self.decoder(encoded) 
        return encoded, decoded