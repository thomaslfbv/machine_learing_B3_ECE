# Classification d’expressions faciales sur FER2013

Projet de machine learning de B3 à l’ECE. À partir d’un portrait en niveaux de gris, le modèle cherche à prédire l’une des sept catégories d’expression du jeu de données [FER2013](https://www.kaggle.com/datasets/msambare/fer2013).

Le projet en est à sa **première phase d’expérimentation**. Le groupe complète le [notebook initial](./Projet.ipynb) avant d’ajouter d’autres notebooks. Le premier modèle sert de point de comparaison pour les essais à venir.

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
jupyter lab Projet.ipynb
```

Dans Jupyter, choisir le noyau de cet environnement et exécuter les cellules dans l’ordre jusqu’à l’affichage de la précision de validation. Le premier lancement nécessite une connexion pour télécharger les images dans `data/`. La dernière cellule de matrice de confusion est un brouillon et peut être ignorée pour reproduire cette phase.

## Organisation proposée pour la suite

| Emplacement | Contenu |
| --- | --- |
| [Projet.ipynb](./Projet.ipynb) | Notebook en cours à la racine. Conserver ici la préparation des données et le premier modèle. |
| `notebooks/` | Futurs essais, avec un numéro et une question précise. Par exemple, `02_analyse_erreurs.ipynb` pour identifier les expressions que le modèle confond. |
| `src/` | Fonctions à extraire dès que deux notebooks réutilisent le même découpage, prétraitement ou calcul de métriques. |
| `data/` | Images téléchargées localement. Le fichier `.gitignore` exclut déjà ce dossier et l’environnement `.venv/`. |
| `results/` | Tableau comparatif des expériences et figures retenues pour le rapport. |

Les chemins actuels du notebook sont relatifs à la racine du dépôt. Pour les futurs notebooks placés dans `notebooks/`, centraliser le chemin vers `data/` afin que tous utilisent les mêmes images et le même découpage.

## Travailler en groupe sur GitHub

1. Ouvrir une *issue* par question expérimentale, par exemple « Quelles expressions le modèle linéaire confond-il le plus souvent ? ». Y préciser le résultat attendu, ici une matrice de confusion sur la validation.
2. Attribuer l’essai à une personne et travailler sur une branche courte, par exemple `exp/analyse-erreurs`. Éviter les modifications simultanées d’un même notebook pour faciliter la fusion.
3. Terminer chaque notebook par les paramètres utilisés, le score de validation et une courte conclusion. Reporter les résultats dans un tableau commun avec un lien vers le notebook correspondant.
4. Ouvrir une *pull request* et demander à un autre membre de relancer le notebook avant la fusion dans `main` (ou la branche principale du dépôt).

## Équipe

Thomas Lefebvre, Junior, Jack, Kemuel, Mory et Adolphe.
