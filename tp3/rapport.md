# TP3 — Réseaux convolutifs

---

## Exercice 1 : Préparation d'ImageNet et augmentation de données

### 1. Que se passe-t-il mathématiquement et numériquement lors de la descente de gradient si l'on oublie l'étape `transforms.Normalize` ? Pourquoi les statistiques (moyenne et écart-type) doivent-elles être calculées uniquement sur le jeu d'entraînement ?

Sans normalisation, les pixels sont dans [0, 1] : tous positifs et non centrés. Comme ∂L/∂wᵢ = δ·xᵢ, tous les gradients d'un neurone ont le même signe, ce qui donne des mises à jour en zig-zag, une surface de perte mal conditionnée et une convergence lente. Les initialisations (Kaiming/Xavier) supposent aussi des entrées centrées réduites, d'où un risque d'explosion ou de disparition des gradients.

Les statistiques viennent du train seul pour éviter une **fuite d'information** : la validation doit simuler des données jamais vues. On applique ensuite ces mêmes statistiques à la validation.

### 2. Pourquoi utilise-t-on `RandomResizedCrop` et `RandomHorizontalFlip` sur le jeu d'entraînement, mais `Resize` + `CenterCrop` (sans flip) sur le jeu de validation ?

- **Train** : les transformations aléatoires diversifient les données et apprennent des invariances (échelle, position, symétrie). C'est une régularisation.
- **Validation** : on mesure, on n'apprend pas. L'évaluation doit être déterministe et reproductible (important pour l'early stopping) et centrée sur l'objet.

### 3. Donnez un exemple de tâche de classification d'images où l'augmentation par `RandomHorizontalFlip` serait néfaste et fausserait les étiquettes.

Toute tâche où l'orientation gauche/droite définit la classe : panneaux « tourner à gauche / à droite », reconnaissance de caractères (b ↔ d), main gauche / main droite.

---

## Exercice 2 : Étude théorique et implémentation des architectures (MLP vs AlexNet)

### 2.a. Calculez à la main le nombre exact de paramètres de ce MLP (en comptant les biais, puis sans les biais). Quelle couche concentre plus de 99 % des paramètres ?

Entrée aplatie : 3 × 227 × 227 = 154 587.

| Couche | Poids | Biais | Total |
|---|---|---|---|
| Linear1 (154 587 → 256) | 39 574 272 | 256 | 39 574 528 |
| Linear2 (256 → 256) | 65 536 | 256 | 65 792 |
| Linear3 (256 → 1000) | 256 000 | 1 000 | 257 000 |
| **Total** | **39 895 808** | 1 512 | **39 897 320** |

- Avec biais : **39 897 320**. Sans biais : **39 895 808**.
- **Linear1** concentre **99,2 %** des paramètres.

### Outre ce nombre élevé de paramètres en entrée, quel est le défaut structurel majeur du MLP lorsqu'on lui fournit une image aplatie (flatten) ?

Le `flatten` détruit la structure spatiale. Le MLP ignore le voisinage des pixels et n'a aucune invariance par translation : un objet décalé active des poids différents, et chaque motif doit être réappris à chaque position.

### 2.b. Recopiez et complétez l'intégralité de ce tableau en format Markdown dans votre rapport. Pour la colonne Mémoire sortie (KB), calculez la mémoire occupée par les activations de sortie de chaque couche pour une seule image en supposant des flottants 32 bits (4 octets, 1 KB = 1024 octets).

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

Formules : H_out = ⌊(H + 2P − K)/S⌋ + 1 ; paramètres conv = C_in·K²·C_out + C_out ; mémoire = C·H·W·4 / 1024.

### Dans AlexNet, où se concentre l'essentiel des paramètres (poids) ? À l'inverse, où se concentre l'essentiel de la mémoire des activations ?

- **Paramètres** : dans les couches denses (58,6 M, soit **96 %**, dont 37,8 M pour Linear1). Les convolutions n'en ont que 2,5 M.
- **Activations** : dans les premières convolutions, où la résolution est grande. Conv1 + Pool1 + Conv2 font ≈ 1 513 KB, soit **65 %**.

### Comparez la mémoire totale des activations d'AlexNet (environ 2340 KB par image) avec celle du MLP de l'étape 1. Expliquez ce paradoxe alors que les deux réseaux ont un ordre de grandeur similaire en nombre de paramètres.

Le MLP n'a que 1 512 activations (**≈ 5,9 KB**), contre ≈ 2 340 KB pour AlexNet (×400).

Une couche dense utilise chaque poids une seule fois : beaucoup de paramètres, peu de sorties. Une convolution **partage ses poids** sur toutes les positions : peu de paramètres, mais de grandes cartes d'activations à stocker pour la backpropagation.

### 2.c. Vérification avec `count_parameters`

![Vérification count_parameters](/images/count_parameters.png)

Les valeurs obtenues (MLP : 39 897 320 ; AlexNet : 61 100 840) correspondent aux calculs théoriques.

---

## Exercice 3 : Boucle d'entraînement, Early Stopping et comparaison rapide

### 1. Quel est le rôle exact de `model.eval()` et `model.train()` vis-à-vis des couches `nn.Dropout` présentes dans AlexNet ? Que se passerait-il si l'on oubliait `model.eval()` pendant l'évaluation ?

- `train()` : le Dropout annule aléatoirement 50 % des activations et multiplie les autres par 1/(1−p).
- `eval()` : le Dropout devient l'identité, le réseau complet est utilisé.

Sans `model.eval()`, l'évaluation serait bruitée et dégradée, ce qui fausserait les métriques et l'early stopping.

### 2. Sur un jeu de données à 1000 classes équilibrées comme ImageNet, quelle est la valeur théorique de l'accuracy et de la CrossEntropyLoss à l'initialisation ? Vos premières époques sont-elles cohérentes avec ces valeurs ?

Théorie : accuracy = 1/1000 = **0,1 %** ; loss = ln(1000) ≈ **6,91**.

| Modèle | Train Loss (ép. 1) | Val Loss (ép. 1) | Val Acc (ép. 1) |
|---|---|---|---|
| AlexNet | 6,908 | 6,907 | 0,14 % |
| AlexNet profond | 6,908 | 6,908 | 0,04 % |
| MLP | 7,193 | 7,011 | 0,18 % |

C'est cohérent. Le MLP démarre un peu au-dessus de 6,91 : avec 154 587 entrées, ses logits initiaux ne sont pas uniformes.

### 3. Comparez les courbes TensorBoard (Train/Val Loss et Accuracy) du MLP et d'AlexNet sur ce sous-ensemble de 20 000 images. Lequel apprend le mieux ? Lequel sur-apprend le plus vite et pourquoi ?

![Logs des runs MLP, AlexNet et AlexNet profond](/images/logs_runs_rapides.png)

![Loss Train - sous-ensemble 20 000 images](/images/loss_train_subset.png)

![Loss Validation - sous-ensemble 20 000 images](/images/loss_val_subset.png)

![Accuracy Train - sous-ensemble 20 000 images](/images/accuracy_train_subset.png)

![Accuracy Validation - sous-ensemble 20 000 images](/images/accuracy_val_subset.png)

| Modèle (ép. 10) | Train Loss | Val Loss | Train Acc | Val Acc |
|---|---|---|---|---|
| MLP | 6,763 | 6,879 | 0,69 % | 0,46 % |
| AlexNet | 6,395 | 6,441 | 0,91 % | 1,12 % |

- **AlexNet apprend le mieux** : sa val loss baisse régulièrement et sa val acc atteint 1,12 % (11× le hasard).
- **Le MLP sur-apprend le plus vite** : sa train acc monte, mais sa val acc plafonne puis recule (0,52 % → 0,46 %) et l'écart train/val est plus grand (0,12 contre 0,05). Avec ~40 M de poids pour 20 images par classe, sans Dropout ni a priori spatial, il mémorise les pixels.
- AlexNet résiste mieux grâce au partage de poids et au Dropout.

### 3.b. Observez les courbes de ce réseau plus profond par rapport à l'AlexNet standard. La difficulté à démarrer l'apprentissage relève-t-elle du sur-apprentissage ou du sous-apprentissage ? Comment l'expliquez-vous du point de vue de la propagation du gradient ?

Courbes : run `alexnet_deep` (orange) sur les captures ci-dessus.

Le réseau profond n'apprend pas : la train loss reste à ≈ 6,89 (≈ ln(1000)), l'accuracy reste au niveau du hasard, et l'early stopping se déclenche à l'époque 6.

C'est du **sous-apprentissage** : même la loss d'entraînement ne baisse pas.

Explication : à chaque couche, la rétropropagation multiplie le gradient par les poids et les dérivées des ReLU. Sans BatchNorm ni connexion résiduelle, le signal s'atténue couche après couche (**vanishing gradient**) : les premières couches n'apprennent presque pas. Remèdes : BatchNorm, connexions résiduelles, initialisation de He.

---

## Exercice 4 : Lancement et analyse d'un entraînement long via sbatch

### 1. Profilage temporel et matériel : combien de temps dure exactement une époque complète sur les 1,28 million d'images d'entraînement avec votre GPU ? À cette vitesse, combien d'heures faudrait-il pour compléter 100 époques sans Early Stopping ?

- GPU NVIDIA L4, batch 256 : **24 min 15 s** d'entraînement par époque (5 005 itérations à 3,44 it/s), soit **≈ 25 min** avec la validation.
- 100 époques ≈ 100 × 25 min ≈ **42 h**.

Le job a été arrêté par la limite de 4 h pendant la 10e époque.

### 2. Analyse de l'entraînement au fil de l'eau : courbes sur le dataset complet et comparaison de l'écart Loss/Train – Loss/Validation avec le sous-ensemble de 20 000 images. Pourquoi le sur-apprentissage est-il beaucoup plus faible ici ?

![Loss Train - ImageNet complet](/images/loss_train_imagenet.png)

![Loss Validation - ImageNet complet](/images/loss_val_imagenet.png)

![Accuracy Train - ImageNet complet](/images/accuracy_train_imagenet.png)

![Accuracy Validation - ImageNet complet](/images/accuracy_val_imagenet.png)

| Run | Époque | Train Loss | Val Loss | Val Acc |
|---|---|---|---|---|
| 20 000 images | 10 | 6,40 | 6,44 | 1,1 % |
| ImageNet complet | 9 | 3,39 | 2,89 | 37,3 % |

- **Sous-ensemble** : la val loss passe au-dessus de la train loss et l'écart grandit. C'est le début du sur-apprentissage.
- **Dataset complet** : les deux courbes descendent en parallèle, la val loss reste sous la train loss. Il n'y a pas de sur-apprentissage.

Avec ≈ 1 280 images par classe (64× plus), et une augmentation qui rend chaque époque différente, le modèle ne peut pas mémoriser : il doit apprendre des motifs généralisables.

### 3. Observation surprenante : lors des premières époques, la perte de validation est souvent inférieure à la perte d'entraînement. Donnez deux raisons techniques liées à notre code.

Observé ici : val loss 2,89 < train loss 3,39 à l'époque 9.

1. **Dropout** : actif en train (prédictions dégradées), désactivé en validation (réseau complet).
2. **Data augmentation** : images recadrées et retournées, plus difficiles en train ; image centrée et « propre » en validation.

De plus, la train loss est moyennée sur toute l'époque, alors que la val loss est mesurée avec le modèle de fin d'époque.