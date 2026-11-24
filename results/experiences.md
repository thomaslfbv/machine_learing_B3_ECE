# Résultats des expériences sur FER2013

Les essais utilisent une régression logistique. Le score indique la proportion de prédictions correctes sur les 4 307 images de validation.

| Essai | Entrée du modèle | Score de validation | Écart avec l’essai 01 | Notebook |
| --- | --- | ---: | ---: | --- |
| 01 | 2 304 pixels par image (48 × 48) | 33,97 % | Référence | [Essai sans PCA](../notebooks/01_Test_Brut_NO_PCA.ipynb) |
| 02 | 100 composantes PCA par image | 37,15 % | +3,18 points | [Essai avec PCA](../notebooks/02_Test_avec_PCA_0.ipynb) |
| 03 | 150 composantes PCA par image (`C = 0,1` retenu par GridSearchCV) | 37,75 % | +3,78 points | [Réglage de la PCA et de C](../notebooks/03_Test_Règler_PCA_et_C.ipynb) |
