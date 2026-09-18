import os, random, datetime, torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Hyperparamètres (faciles à modifier pour nos futures expériences)
hparams = dict(model="MLP", batch_size=32, lr=1e-2, seed=0, weight_decay=0.0)

# 1. Création d'un nom de dossier unique (modèle + hparams + timestamp)
run_name = f"{hparams['model']}/bs{hparams['batch_size']}_lr{hparams['lr']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
logdir = os.path.join("runs", run_name)
print("Logdir:", logdir)

# Instanciation du SummaryWriter
writer = SummaryWriter(log_dir=logdir)

# 2. Séparation du jeu de données (on suppose 'trainset' déjà chargé comme à l'exercice précédent)
N = len(trainset)
val_size = int(0.1 * N)
train_size = N - val_size

# Utilisation de random_split avec une graine fixe pour la reproductibilité
train_subset, val_subset = random_split(trainset, [train_size, val_size], generator=torch.Generator().manual_seed(0))

# Création des DataLoaders (à compléter avec hparams)
trainloader = DataLoader(train_subset, batch_size=hparams["batch_size"], shuffle=True, pin_memory=True)
valloader   = DataLoader(val_subset,   batch_size=hparams["batch_size"], shuffle=False, pin_memory=True)

@torch.no_grad()
def epoch_metrics(loader, model, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        logits = model(x)
        loss = criterion(logits, y)
        
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
        
    loss_avg = loss_sum / total
    acc = correct / total
    return loss_avg, acc

# Initialisation (modèle, optimizer, criterion...) identiques à l'exercice précédent.
global_step = 0
EPOCHS = 10

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss_sum, running_total = 0.0, 0

    for b, (x, y) in enumerate(trainloader):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Logging batch (toutes les 10 itérations pour ne pas surcharger)
        if b % 10 == 0:
            writer.add_scalar("Loss/train_step", loss.item(), global_step)

        optimizer.step()
        running_loss_sum += loss.item() * y.size(0)
        running_total    += y.size(0)
        global_step += 1

    # Métriques de fin d'époque
    train_loss = running_loss_sum / running_total
    val_loss, val_acc = epoch_metrics(valloader, model, criterion, device)

    # Logging époque
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val",   val_loss, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)

    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

# Fin d'entraînement
writer.close() # Force l'écriture des données sur le disque