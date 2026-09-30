import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import CardioDataset

# --- Données ---
dataset = CardioDataset("data/cardio_train.csv")
generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(dataset, [0.8, 0.1, 0.1], generator=generator)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

batch = next(iter(train_loader))

# --- Modèle ---
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1), 
            nn.Sigmoid() # Sortie binaire [0, 1]
        )

    def forward(self, x):
        return self.net(x)

# Initialisation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_size=batch["features"].shape[1], hidden_size=128).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 1e-4
l2_lambda = 1e-3

def evaluate(loader):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for b in loader:
            x, y = b["features"].to(device), b["labels"].to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            correct += ((out > 0.5).float() == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total

# --- Entraînement ---
for epoch in range(10):
    model.train()
    for b in train_loader:
        inputs, targets = b["features"].to(device), b["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        base_loss = criterion(outputs, targets)

# Calcul de la pénalité L1 (somme des valeurs absolues des poids)
        l1_penalty = sum(p.abs().sum() for p in model.parameters())

# Calcul de la pénalité L2 (somme des carrés des poids)
        l2_penalty = sum(p.pow(2).sum() for p in model.parameters())

# Loss totale
        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty
        loss.backward()
        optimizer.step()

    train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1:2d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
          f"| val loss {val_loss:.4f} acc {val_acc:.3f}")