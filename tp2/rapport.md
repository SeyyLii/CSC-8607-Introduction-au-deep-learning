# TP2 — Régularisation, optimisation et métriques

---

## Exercice 1 : Création d'un dataset personnalisé

**Dans le code fourni, nous utilisons StandardScaler() sur l'ensemble du dataset (avant le split). Pourquoi est-ce une mauvaise pratique en apprentissage automatique (concept de data leakage) ?**

Normaliser avant le split est un data leakage car la normalisation(la moyenne / variance) incluent val et test, donc des informations du test servent à l'entraînement. Les performances en validation sont alors trop optimistes et ne reflètent pas forcement d'une bonne généralisation.

**Si le dataset était trop volumineux pour tenir dans la RAM (ex: 500 Go de données tabulaires), quelle classe PyTorch utiliseriez-vous à la place de Dataset ?**

On utilise torch.utils.data.IterableDataset, qui lit les données en flux par chunks ou shards via __iter__ au lieu de tout charger. Dans ce cas, le mélange et le split sont à gérer soi-même, car random_split et shuffle=True ne sont pas utilisables. ( A MODIF )


## Exercice 2 : MLP et Régularisation L1 / L2

Lancons d'abord avec l1_lambda = 1e-4 l2_lambda = 1e-3
![alt text](/images/statdebase.png)
**Réglez l1_lambda = 0.1 et l2_lambda = 0. Qu'observez-vous sur l'apprentissage (loss et précision) ?** 
**Pourquoi une régularisation trop forte produit-elle cet effet (comment appelle-t-on ce phénomène) ?**

![alt text](/images/underfitting.png)
Avec les valeurs de base (L1 = 1e-4, L2 = 1e-3), le modèle apprend : la loss de validation passe de 0,631 à 0,584 et la précision de 66 % à 70 % en 10 époques.

Avec L1 = 0,1, l'apprentissage est bloqué dès la première époque : la loss reste à 0,6931 et la précision à 50 % (aucun apprentissage). C'est du sous-apprentissage (underfitting).

**Dans PyTorch, il n'est généralement pas nécessaire d'ajouter la pénalité L2 à la main dans la boucle. Quel argument de l'optimiseur (ex: optim.SGD) permet d'appliquer cette régularisation L2 automatiquement ?**

C'est l'argument weight_decay, par exemple optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3).

**Quelle est la différence conceptuelle sur les poids du réseau entre une régularisation L1 et L2 ?**

L1 pousse beaucoup de poids exactement à zéro, ce qui revient à une sélection de features.
L2 réduit tous les poids proportionnellement à leur valeur, sans les annuler : les poids restent petits et répartis.


## Exercice 3 : Comparaison des Optimiseurs et TensorBoard

![alt text](/images/SGDvsMOMENTUM.png)

**Quel optimiseur converge le plus rapidement initialement ?**

Adam converge le plus vite au départ : dès la première époque, sa loss (≈ 0,59) est la plus basse, juste devant RMSprop (≈ 0,60), et loin devant Momentum (≈ 0,66) et SGD (≈ 0,69). Les deux optimiseurs adaptatifs restent ensuite au coude à coude et finissent autour de 0,54 (RMSprop 0,536, Adam 0,540).

**Comparez la courbe de SGD simple et Momentum. Quel est l'effet de l'ajout du moment sur la descente de gradient ?**

Le moment accélère nettement la descente : après 30 époques, Momentum atteint une loss de 0,571 contre 0,626 pour SGD simple, qui descend lentement et de façon presque linéaire. En accumulant les gradients passés, le moment augmente le pas effectif dans les directions stables et amortit les oscillations, ce qui accélère la convergence.

### Exercice 4 : Analyse des métriques (Précision, Rappel, F1, AUC)

**Rappelez la définition de la précision (Precision) et du rappel (Recall).**
La précision est la part des prédictions positives qui sont correctes : TP / (TP + FP). Le rappel est la part des vrais positifs que le modèle détecte : TP / (TP + FN).


**Dans le contexte médical (détecter une maladie cardiovasculaire), vaut-il mieux privilégier un modèle avec une forte Précision ou un modèle avec un fort Rappel ? Justifiez votre réponse.**

Il est préférable de privilegier le rappel car, dans un contexte médical, on préfèrera rater le moins de malades possibles quitte à avoir plus de fausses alertes : un faux négatif sera un malade non détecté et donc non soigné, là ou un faux positif entraînera uniquement des examens complémentaires.

**À quoi sert l'aire sous la courbe ROC (AUC) par rapport aux autres métriques calculées à un seuil fixe de 0.5 ?**

L'AUC mesure la capacité du modèle à classer les positifs au-dessus des négatifs sur tous les seuils possibles, indépendamment du seuil 0,5. Elle évalue la qualité intrinsèque du modèle, alors que précision, rappel et F1 dépendent du seuil choisi, qu'on peut ensuite ajuster, par exemple l'abaisser pour augmenter le rappel.


Voici les métriques du meilleur modèle (RMSprop, vu en ex3) :
Precision: 0.7593 | Recall: 0.6933 | F1: 0.7248 | AUC: 0.8036

L'AUC de 0,80 montre que le modèle distingue correctement les malades des non-malades. En revanche, le rappel (0,69) est inférieur à la précision (0,76) : le modèle rate environ 31 % des patients malades, ce qui est problématique en contexte médical. Pour privilégier le rappel, on pourrait abaisser le seuil de décision sous 0,5, au prix d'une baisse de la précision.
