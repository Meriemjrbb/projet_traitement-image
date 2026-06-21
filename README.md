# Mini-projet : Détection et Comptage d'Objets

Ce projet Python réalise un traitement d'images simple pour détecter des objets dans une image, les segmenter et compter leur nombre.

## Contenu du projet

- `mini_projet.py` : script principal qui charge une image, la convertit en niveaux de gris, lisse le bruit, applique un seuillage automatique par Otsu, nettoie l'image par morphologie, détecte les objets par composantes connexes, puis enregistre le résultat.

- `resultats/` : dossier de sortie où sont enregistrées les images intermédiaires et le résultat final.
- `image.png`, `image1.png`, `image2.png`, `image3.png` : exemples d'images présentes dans le dossier.
- fichiers de rapport Word/PDF.

## Prérequis

- Python 3.x
- Pillow

## Installation

1. Ouvre un terminal dans le dossier du projet.
2. Installe Pillow :

```bash
pip install pillow
```

## Utilisation

Le script `mini_projet.py` contient un chemin d'image prédéfini dans la variable `CHEMIN_IMAGE`.

1. Modifie `CHEMIN_IMAGE` dans `mini_projet.py` pour pointer vers l'image souhaitée si nécessaire.
2. Exécute le script :

```bash
python mini_projet.py
```

## Résultats

Le script enregistre les images suivantes dans `resultats/` :

- `01_niveaux_de_gris.png`
- `02_lissage.png`
- `03_binaire.png`
- `04_morphologie.png`
- `05_detection_finale.png`

Le terminal affiche aussi le nombre d'objets détectés et les coordonnées des bounding boxes.

## Remarques

- Le script détecte les objets sombres sur un fond clair.
- Si tu veux tester une autre image, adapte le chemin du fichier dans `mini_projet.py`.
- `mini_projet copy.py` est une copie du même code et peut servir de sauvegarde.
