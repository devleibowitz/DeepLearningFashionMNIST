import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ===== Define a simple Transformer encoder model =====
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=2, seq_len=512):
        super(SimpleTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)                   # (batch, seq_len, model_dim)
        x = self.encoder(x)                      # (batch, seq_len, model_dim)
        x = x.mean(dim=1)                        # mean pooling over sequence
        return self.classifier(x)                # (batch, 1)

# ===== Generate fake CTG signal data =====
batch_size = 32
seq_len = 512
input_dim = 2

X_fake = torch.randn(100, seq_len, input_dim)  # 100 fake samples
y_fake = torch.randint(0, 2, (100, 1)).float()

# ===== Wrap in DataLoader =====
dataset = TensorDataset(X_fake, y_fake)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ===== Instantiate model, loss, optimizer =====
model = SimpleTransformer(input_dim=input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===== Train for a few epochs =====
model.train()
for epoch in range(3):
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

print("Finished training.")

# ===== Save model to disk =====
torch.save(model, 'transformer.pth')
print("Saved model to transformer.pth")