# TP3 — Réseaux convolutifs

---

## Exercice 1

### **1. Que se passe-t-il mathématiquement et numériquement lors de la descente de gradient si l'on oublie l'étape transforms.Normalize ? Pourquoi les statistiques (moyenne et écart-type) doivent-elles être calculées uniquement sur le jeu d'entraînement ?**

Les pixels sont dans [0, 1], donc tous positifs et non centrés. Pour un neurone, ∂L/∂wᵢ = δ·xᵢ :
tous les gradients ont le même signe, ce qui produit des mises à jour en zig-zag. La surface de
perte est mal conditionnée et la convergence plus lente. De plus, les initialisations
(Kaiming/Xavier) supposent des entrées centrées réduites : sans normalisation, on risque
l'explosion ou la disparition des gradients.

Les statistiques sont calculées sur le train uniquement pour éviter une **fuite d'information** :
la validation doit simuler des données jamais vues. On applique ensuite la même normalisation
(stats du train) à la validation.

### **2. Pourquoi utilise-t-on RandomResizedCrop et RandomHorizontalFlip sur le jeu d'entraînement, mais Resize + CenterCrop (sans flip) sur le jeu de validation ?**
- **Train** : crops et flips aléatoires diversifient les données et apprennent des invariances
  (échelle, position, symétrie). C'est une régularisation contre l'overfitting.
- **Validation** : on mesure, on n'apprend pas. `Resize` + `CenterCrop` donne une évaluation
  reproductible et non bruitée (important pour l'early stopping), centrée sur l'objet, à une
  échelle proche de l'entraînement.

### **3. Donnez un exemple de tâche de classification d'images où l'augmentation par RandomHorizontalFlip serait néfaste et fausserait les étiquettes.**

Quand l'orientation gauche/droite définit la classe : panneaux « tourner à gauche/droite »,
reconnaissance de caractères (b ↔ d), main gauche/droite.


## Exercice 2 : Étude théorique et implémentation des architectures (MLP vs AlexNet) 

### **Calculez à la main le nombre exact de paramètres de ce MLP (en comptant les biais, puis sans les biais). Quelle couche concentre plus de 99% des paramètres ?**
Entrée : 3×227×227 = 154 587.

| Couche | Poids | Biais | Total |
|---|---|---|---|
| Linear1 (154 587 → 256) | 39 574 272 | 256 | 39 574 528 |
| Linear2 (256 → 256) | 65 536 | 256 | 65 792 |
| Linear3 (256 → 1000) | 256 000 | 1 000 | 257 000 |
| **Total** | **39 895 808** | 1 512 | **39 897 320** |

Linear1 concentre **99,2 %** des paramètres.

### **Outre ce nombre élevé de paramètres en entrée, quel est le défaut structurel majeur du MLP lorsqu'on lui fournit une image aplatie (flatten) ?**

**Défaut structurel** : le `flatten` détruit la structure spatiale. Le MLP ignore la proximité
des pixels et n'a aucune invariance par translation : un objet décalé de quelques pixels active
des poids complètement différents, et chaque motif doit être réappris à chaque position.

### **Recopiez et complétez l'intégralité de ce tableau en format Markdown dans votre rapport. Pour la colonne Mémoire sortie (KB), calculez la mémoire occupée par les activations de sortie de chaque couche pour une seule image en supposant des flottants 32 bits (4 octets, $1\text{ KB} = 1024\text{ octets}$).**

| Couche | Entrée (C, H×W) | Hyperparamètres | Sortie (C, H×W) | # Paramètres | Mémoire sortie (KB) |
|---|---|---|---|---|---|
| Conv1 + ReLU | 3, 227×227 | 64, 11×11, S=4, P=2 | 64, 56×56 | 23 296 | 784 |
| MaxPool1 | 64, 56×56 | 3×3, S=2, P=0 | 64, 27×27 | 0 | 182,25 |
| Conv2 + ReLU | 64, 27×27 | 192, 5×5, S=1, P=2 | 192, 27×27 | 307 392 | 546,75 |
| MaxPool2 | 192, 27×27 | 3×3, S=2, P=0 | 192, 13×13 | 0 | 126,75 |
| Conv3 + ReLU | 192, 13×13 | 384, 3×3, S=1, P=1 | 384, 13×13 | 663 936 | 253,5 |
| Conv4 + ReLU | 384, 13×13 | 256, 3×3, S=1, P=1 | 256, 13×13 | 884 992 | 169 |
| Conv5 + ReLU | 256, 13×13 | 256, 3×3, S=1, P=1 | 256, 13×13 | 590 080 | 169 |
| MaxPool3 | 256, 13×13 | 3×3, S=2, P=0 | 256, 6×6 | 0 | 36 |
| Flatten + Dropout | 256, 6×6 | p = 0,5 | 9 216 (1D) | 0 | 36 |
| Linear1 + ReLU + Dropout | 9 216 | 4096, p = 0,5 | 4 096 | 37 752 832 | 16 |
| Linear2 + ReLU | 4 096 | 4096 | 4 096 | 16 781 312 | 16 |
| Linear3 (Sortie) | 4 096 | 1000 | 1 000 | 4 097 000 | 3,91 |
| **Total** | | | | **61 100 840** | **≈ 2 339** |

Formules : H_out = ⌊(H + 2P − K)/S⌋ + 1 ; params conv = C_in·K²·C_out + C_out ;
mémoire = C·H·W·4 / 1024.

### **Dans AlexNet, où se concentre l'essentiel des paramètres (poids) ? À l'inverse, où se concentre l'essentiel de la mémoire des activations (indispensable à stocker lors du forward pour calculer la backpropagation) ?**
- **Paramètres** : dans les couches denses (58,6 M, soit **96 %**). Linear1 seule en a 37,8 M.
  Les convolutions n'en ont que 2,5 M.
- **Activations** : dans les premières convolutions, où la résolution spatiale est grande.
  Conv1 + Pool1 + Conv2 font ≈ 1 513 KB, soit **65 %**. Les couches denses font moins de 2 %.

### **Comparez la mémoire totale des activations d'AlexNet (environ $2340\text{ KB}$ par image) avec celle du MLP de l'étape 1. Expliquez ce paradoxe alors que les deux réseaux ont un ordre de grandeur similaire en nombre de paramètres.**
Le MLP n'a que 256 + 256 + 1000 = 1 512 activations, soit **≈ 5,9 KB** par image, contre
≈ 2 340 KB pour AlexNet (×400), alors que les deux modèles ont des dizaines de millions de
paramètres.

Explication : une couche dense utilise chaque poids **une seule fois**. Beaucoup de paramètres
donnent peu de sorties. Une convolution **partage ses poids** : le même petit filtre est
appliqué à chaque position spatiale. Peu de paramètres produisent donc des cartes de
caractéristiques volumineuses, à stocker pour la backpropagation. La VRAM d'un CNN est dominée
par les activations, celle d'un MLP par les poids.