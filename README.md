# Classification d’expressions faciales sur FER2013

Projet de machine learning de B3 à l’ECE. À partir d’un portrait en niveaux de gris, le modèle cherche à prédire l’une des sept catégories d’expression du jeu de données [FER2013](https://www.kaggle.com/datasets/msambare/fer2013).

Le projet en est à sa **première phase d’expérimentation**. Le groupe complète le [notebook initial](./notebooks/01_Test_Brut_NO_PCA.ipynb) avant d’ajouter d’autres notebooks. Le premier modèle sert de point de comparaison pour les essais à venir.

## Première expérience

Les images sont redimensionnées à 48 × 48 pixels. Le notebook réserve 15 % du dossier d’entraînement à la validation avec une graine de découpage fixée à 42. Le dossier de test fourni avec FER2013 est gardé pour l’évaluation finale.

| Sous-ensemble | Images | Usage |
| --- | ---: | --- |
| Entraînement | 24 402 | Apprendre les paramètres du modèle |
| Validation | 4 307 | Comparer les expériences |
| Test | 7 178 | Évaluer le modèle retenu en fin de projet |

La régression logistique multiclasse aplatit chaque image en 2 304 valeurs et produit un score par catégorie. L’entraînement utilise PyTorch, une perte d’entropie croisée et l’optimiseur Adam avec un taux d’apprentissage de 0,001, pendant 20 époques, par lots de 64 images.

**Résultat provisoire** — La sortie enregistrée affiche **33,97 % de bonnes prédictions sur la validation**. La matrice de confusion reste à compléter. L’évaluation finale sur le jeu de test reste à faire.

## Exécuter le notebook

Depuis la racine du dépôt, dans un terminal macOS ou Linux, préparer l’environnement et ouvrir Jupyter avec les commandes suivantes.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab kagglehub torch torchvision scikit-learn
jupyter lab ## Le nom du notebook à ouvrir
```

Dans Jupyter, choisir le noyau de cet environnement et exécuter les cellules dans l’ordre jusqu’à l’affichage de la précision de validation. Le premier lancement nécessite une connexion pour télécharger les images dans `data/`. La dernière cellule de matrice de confusion est un brouillon et peut être ignorée pour reproduire cette phase.



## Équipe

Thomas Lefebvre, Junior, Jack, Kemuel, Mory et Adolphe.
