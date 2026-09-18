# TP1 — Prise en main du cluster Albator (SLURM) et de l'environnement Deep Learning

**Auteur :** Ilyes BELKHIR

---

## 1. Connexion au cluster

Connexion en SSH au nœud de connexion `arcadia-slurm-controller` du cluster HPC Albator (Direction de l'Enseignement, Télécom SudParis).

![Connexion au cluster Albator](images/01_connexion_albator.png)

---

## 2. Accès à un GPU en mode interactif

### 2.1 `nvidia-smi` sur la machine de connexion

Sur le nœud de connexion, la commande échoue comme prévu : `nvidia-smi` n'est même pas installé, car cette machine ne dispose d'aucun GPU. Elle sert uniquement à se connecter et à soumettre des jobs.

J'ai ensuite demandé des ressources en mode interactif :

```bash
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --cpus-per-task=1 --mem=8G --pty bash
```

SLURM m'a attribué le job **1540** sur le nœud `starfighter-slurm-node-03-1`.

![nvidia-smi sur le contrôleur puis srun](images/02_nvidia_smi_controller_srun.png)

### 2.2 `nvidia-smi` sur le nœud de calcul

> **Quel est le modèle exact du GPU alloué ?**

Le GPU alloué est une **NVIDIA L4**, avec **23 034 MiB** de mémoire (environ 24 Go). Le driver est en version 595.84 et supporte CUDA jusqu'à la version 13.2. Au moment de la capture, le GPU était au repos (état P8, 0 % d'utilisation, aucun processus).

![nvidia-smi sur le nœud de calcul](images/03_nvidia_smi_noeud.png)

---

## 3. Suivi et annulation d'un job

Depuis un second terminal connecté au contrôleur, j'ai listé mes jobs :

```bash
squeue -u $USER
```

![squeue](images/04_squeue.png)

Le job interactif a le JobID **1540**, il est dans l'état `R` (*running*) sur `starfighter-slurm-node-03-1`.

> **Commande exacte utilisée pour annuler le job :**

```bash
scancel 1540
```

![scancel](images/05_scancel.png)

Juste après l'annulation, le job passe dans l'état `CG` (*completing*) le temps que SLURM libère les ressources, puis il disparaît de la file d'attente.

---

## 4. Partitions disponibles

```bash
sinfo -s
```

![sinfo](images/06_sinfo.png)

Quatre partitions sont visibles : `admin`, `arcadia`, `darkshadow` et `gpu`. La partition `gpu`, utilisée dans ce TP, regroupe 18 nœuds (3 alloués et 15 libres au moment de la commande) avec une durée maximale de **12 h** par job. La colonne `NODES(A/I/O/T)` se lit : alloués / inactifs (*idle*) / autres / total.

---

## 5. Soumission d'un job batch

Le script `hello.sh` (ajouté au dépôt Git dans le dossier `TP1`) :

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH -J hello-slurm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

echo "Job $SLURM_JOB_ID on $SLURM_NODELIST"
nvidia-smi || echo "nvidia-smi indisponible"
echo "Bonjour depuis SLURM !"
```

Soumission :

```bash
sbatch hello.sh
```

![sbatch](images/07_sbatch_hello.png)

> **Quel est le nom exact du fichier de log généré ?**

Le fichier de sortie est **`logs/hello-slurm-1541.out`**. Son nom suit le motif `%x-%j.out`, où `%x` est le nom du job (`hello-slurm`, défini par `-J`) et `%j` son identifiant (`1541`). Les erreurs sont redirigées de la même façon vers `logs/hello-slurm-1541.err`.

![Contenu du log](images/08_log_hello.png)

Le log confirme que le job s'est exécuté sur `starfighter-slurm-node-03-1`, qu'il a bien eu accès au GPU L4, et qu'il affiche le message final « Bonjour depuis SLURM ! ».

*Remarque :* SLURM ouvre les fichiers `-o` et `-e` **avant** d'exécuter le script. Le `mkdir -p logs` à l'intérieur du script arrive donc trop tard : si le dossier `logs/` n'existe pas au moment du `sbatch`, le job échoue sans produire de log. Il faut le créer au préalable.

---

## 6. Historique et consommation mémoire

```bash
sacct -j 1541 --format=JobID,State,Elapsed,MaxRSS,ReqMem,ReqCPUS
```

![sacct](images/09_sacct.png)

> **Différence entre `ReqMem` et `MaxRSS` :**

`ReqMem` est la mémoire **réservée** pour le job, c'est-à-dire ce que j'ai demandé à SLURM avec `#SBATCH --mem=8G` dans `hello.sh` : ici 8 Go. `MaxRSS` est la mémoire **réellement utilisée**, c'est-à-dire le pic de mémoire vive occupée par le processus pendant son exécution : ici 17 904 Ko, soit environ 17,5 Mo pour l'étape `1541.batch`. L'écart montre que j'ai réservé bien plus que nécessaire. Sur un cluster partagé, comparer ces deux valeurs permet d'ajuster ses demandes pour ne pas bloquer inutilement des ressources.

---

## 7. Environnement Python `deeplearning`

Après création de l'environnement (Python 3.10), je l'ai activé sur un nœud de calcul (job interactif 1544 sur `starfighter-slurm-node-02-1`) :

```bash
mamba activate deeplearning
```

> **Commandes pour vérifier la version de Python et le chemin du binaire :**

```bash
python --version
# Python 3.10.21

which python
# /mnt/hdd/homes/ibelkhir/miniforge3/envs/deeplearning/bin/python

echo $CONDA_DEFAULT_ENV
# deeplearning
```

![Environnement deeplearning](images/10_env_deeplearning.png)

Le chemin du binaire pointe bien dans `envs/deeplearning`, ce qui confirme que c'est le Python de l'environnement qui est utilisé et non celui du système.

---

## 8. Vérification de l'accès au GPU depuis PyTorch

Script `check_gpu.py` :

```python
import torch

print("PyTorch version:", torch.__version__)
gpu_available = torch.cuda.is_available()
print("CUDA available:", gpu_available)

if gpu_available:
    print("Device count:", torch.cuda.device_count())
    print("Device 0 name:", torch.cuda.get_device_name(0))
else:
    print("Attention, aucun GPU détecté !")
```

> **Sortie du script :**

```
PyTorch version: 2.13.0
CUDA available: False
Attention, aucun GPU détecté !
```

![check_gpu.py](images/11_check_gpu.png)

> **Deux raisons possibles au `CUDA available: False` :**

`torch.cuda.is_available()` ne renvoie `True` que si trois conditions sont réunies : PyTorch est compilé avec le support CUDA, un driver NVIDIA fonctionnel est présent, et au moins un GPU est visible pour le processus. Les causes possibles sont donc :

1. **Version CPU de PyTorch.** Le paquet installé peut être une build sans support CUDA (par exemple si le solveur a choisi la variante CPU). On le vérifie avec `python -c "import torch; print(torch.version.cuda)"` : `None` signifie que PyTorch n'a pas été compilé avec CUDA.
2. **Aucun GPU alloué au job.** Si la session interactive a été lancée sans `--gres=gpu:1` (ou sur un nœud sans GPU), SLURM ne rend aucun GPU visible au processus, même si la machine en possède. On le vérifie en lançant `nvidia-smi` ou `echo $CUDA_VISIBLE_DEVICES` dans le job.

Une troisième cause possible est une incompatibilité entre la version CUDA de la build PyTorch et le driver du nœud, mais elle est peu probable ici car le driver 595.84 supporte CUDA 13.2.

---

## 9. Fichier `environment.yml`

Le fichier suivant a été ajouté au dossier `TP1` du dépôt Git et commité, pour permettre de recréer l'environnement à l'identique (`mamba env create -f environment.yml`) :

```yaml
name: deeplearning
channels:
  - conda-forge
  - pytorch
dependencies:
  - pip
  - python=3.10
  - pytorch
  - pytorch-cuda=12.1
  - tensorboard
  - torchaudio
  - torchvision
```
