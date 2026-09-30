from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
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
            nn.Linear(hidden_size, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Entraînement ---
def train_model(opt_name, learning_rate=0.001, epochs=30):
    model = MLP(input_size=batch["features"].shape[1], hidden_size=128).to(device)
    criterion = nn.BCELoss()

    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif opt_name == "Momentum":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    writer = SummaryWriter(f"runs/cardio_{opt_name}_lr{learning_rate}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for b in train_loader:
            inputs, targets = b["features"].to(device), b["labels"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        writer.add_scalar("Training Loss", avg_loss, epoch)
        print(f"{opt_name} | epoch {epoch+1:2d} | loss {avg_loss:.4f}")

    writer.close()

    return model  # Retourner le modèle entraîné pour l'évaluation

def evaluate_model(model, test_loader):
    model.eval()
    all_targets = []
    all_preds_probs = []
    
    with torch.no_grad(): # Désactiver le calcul des gradients
        for batch in test_loader:
            inputs, targets = batch["features"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            
            # Stocker les probabilités et les cibles pour sklearn
            all_targets.extend(targets.cpu().numpy())
            all_preds_probs.extend(outputs.cpu().numpy())
            
    # Conversion en numpy array
    all_targets = np.array(all_targets)
    all_preds_probs = np.array(all_preds_probs)
    
    # Prédictions binaires (seuil à 0.5)
    all_preds_classes = (all_preds_probs > 0.5).astype(int)
    
    # Calcul des métriques
    precision = precision_score(all_targets, all_preds_classes)
    recall = recall_score(all_targets, all_preds_classes)
    f1 = f1_score(all_targets, all_preds_classes)
    auc = roc_auc_score(all_targets, all_preds_probs)
    
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# --- Expériences ---
for opt in ["SGD", "Momentum", "RMSprop", "Adam"]:
    train_model(opt, learning_rate=0.001)

best_model = train_model("RMSprop", learning_rate=0.001)
evaluate_model(best_model, test_loader)