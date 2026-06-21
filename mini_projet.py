"""
Mini-Projet : Système de détection et comptage d'objets dans une image
Module      : Traitement d'Images
Description : Application Python qui détecte automatiquement des objets
              dans une image, les segmente et compte leur nombre.
"""

from PIL import Image, ImageDraw
import os
import sys


# ============================================================
# ÉTAPE 1 : Chargement de l'image
# ============================================================

def charger_image(chemin):
    try:
        photo = Image.open(chemin)
        print(f"  Image     : {chemin}")
        print(f"  Format    : {photo.format}")
        print(f"  Taille    : {photo.size}")
        print(f"  Mode      : {photo.mode}")
        return photo
    except FileNotFoundError:
        print(f"Erreur : fichier introuvable -> {chemin}")
        sys.exit(1)


# ============================================================
# ÉTAPE 2 : Conversion en niveaux de gris
# ============================================================

def convertir_gris(photo):
    """Formule de luminance perceptuelle : 0.299R + 0.587G + 0.114B"""
    dim_x, dim_y = photo.size
    gris = Image.new('L', (dim_x, dim_y))
    for y in range(dim_y):
        for x in range(dim_x):
            p = photo.getpixel((x, y))
            r, g, b = (p[0], p[1], p[2]) if isinstance(p, (tuple, list)) else (p, p, p)
            gris.putpixel((x, y), int(0.299 * r + 0.587 * g + 0.114 * b))
    return gris

# ============================================================
# ÉTAPE 3 : Lissage 3x3 (réduction du bruit)
# ============================================================

def lisser(image):
    """Filtre moyenneur 3x3 pour atténuer le bruit."""
    dim_x, dim_y = image.size
    dst = Image.new('L', (dim_x, dim_y))
    for y in range(1, dim_y - 1):
        for x in range(1, dim_x - 1):
            s = sum(image.getpixel((x + dx, y + dy))
                    for dy in range(-1, 2) for dx in range(-1, 2))
            dst.putpixel((x, y), s // 9)
    return dst

# ============================================================
# ÉTAPE 4 : Seuillage automatique — méthode d'Otsu
# ============================================================

def seuil_otsu(image):
    """Calcule le seuil optimal qui maximise la variance inter-classe."""
    dim_x, dim_y = image.size
    total = dim_x * dim_y

    histo = [0] * 256
    for y in range(dim_y):
        for x in range(dim_x):
            histo[image.getpixel((x, y))] += 1

    prob = [h / total for h in histo]
    somme_tot = sum(i * prob[i] for i in range(256))

    poids_bg = somme_bg = meilleure_var = 0.0
    meilleur_t = 0

    for t in range(256):
        poids_bg += prob[t]
        somme_bg  += t * prob[t]
        if poids_bg == 0 or poids_bg == 1:
            continue
        poids_fg = 1.0 - poids_bg
        moy_bg   = somme_bg / poids_bg
        moy_fg   = (somme_tot - somme_bg) / poids_fg
        var      = poids_bg * poids_fg * (moy_bg - moy_fg) ** 2
        if var > meilleure_var:
            meilleure_var = var
            meilleur_t = t

    return meilleur_t


def binariser(image, seuil):
    """
    Binarisation : objets sombres sur fond clair.
    Les pixels SOUS le seuil (sombres = objets) deviennent blancs (255).
    Les pixels AU-DESSUS du seuil (clairs = fond) deviennent noirs (0).
    """
    dim_x, dim_y = image.size
    bin_img = Image.new('L', (dim_x, dim_y))
    for y in range(dim_y):
        for x in range(dim_x):
            val = image.getpixel((x, y))
            bin_img.putpixel((x, y), 255 if val < seuil else 0)
    return bin_img


# ============================================================
# ÉTAPE 5 : Morphologie mathématique
# ============================================================

def eroder(image):
    """Érosion 3x3 : chaque pixel = min du voisinage."""
    dim_x, dim_y = image.size
    dst = Image.new('L', (dim_x, dim_y))
    for y in range(1, dim_y - 1):
        for x in range(1, dim_x - 1):
            voisins = [image.getpixel((x + dx, y + dy))
                       for dy in range(-1, 2) for dx in range(-1, 2)]
            dst.putpixel((x, y), min(voisins))
    return dst


def dilater(image):
    """Dilatation 3x3 : chaque pixel = max du voisinage."""
    dim_x, dim_y = image.size
    dst = Image.new('L', (dim_x, dim_y))
    for y in range(1, dim_y - 1):
        for x in range(1, dim_x - 1):
            voisins = [image.getpixel((x + dx, y + dy))
                       for dy in range(-1, 2) for dx in range(-1, 2)]
            dst.putpixel((x, y), max(voisins))
    return dst


def ouverture(image):
    """Érosion puis dilatation : supprime le bruit résiduel."""
    return dilater(eroder(image))


def fermeture(image):
    """Dilatation puis érosion : comble les trous dans les objets."""
    return eroder(dilater(image))


def nettoyer(image):
    """
    Applique 3 fermetures pour combler les trous (écrous, rondelles)
    puis une ouverture pour supprimer le bruit résiduel.
    """
    img = fermeture(fermeture(fermeture(image)))
    img = ouverture(img)
    return img


# ============================================================
# ÉTAPE 6 : Détection par composantes connexes (BFS)
#           + comptage + bounding boxes
# ============================================================

COULEURS = [
    (220, 60,  60),  (60,  180, 60),  (60,  100, 220),
    (220, 170, 40),  (170, 60,  220), (40,  200, 200),
    (220, 110, 40),  (100, 220, 80),  (220, 60,  170),
    (80,  150, 220), (180, 200, 60),
]


def detecter_objets(image_bin, taille_min=300):
    """
    Étiquetage des composantes connexes par BFS (4-connexité).
    taille_min : taille minimale en pixels pour être considéré comme un objet.
    """
    dim_x, dim_y = image_bin.size
    visite = [[False] * dim_y for _ in range(dim_x)]
    resultat = image_bin.convert('RGB')
    objets = []

    for sy in range(dim_y):
        for sx in range(dim_x):
            if image_bin.getpixel((sx, sy)) == 255 and not visite[sx][sy]:
                file   = [(sx, sy)]
                visite[sx][sy] = True
                pixels = []

                while file:
                    cx, cy = file.pop(0)
                    pixels.append((cx, cy))
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < dim_x and 0 <= ny < dim_y:
                            if not visite[nx][ny] and image_bin.getpixel((nx, ny)) == 255:
                                visite[nx][ny] = True
                                file.append((nx, ny))

                if len(pixels) >= taille_min:
                    couleur = COULEURS[len(objets) % len(COULEURS)]
                    for px, py in pixels:
                        resultat.putpixel((px, py), couleur)
                    xs = [p[0] for p in pixels]
                    ys = [p[1] for p in pixels]
                    objets.append({
                        'id':     len(objets) + 1,
                        'x_min':  min(xs), 'y_min': min(ys),
                        'x_max':  max(xs), 'y_max': max(ys),
                        'taille': len(pixels)
                    })

    return resultat, len(objets), objets


def dessiner_boites(image, objets):
    """Dessine les bounding boxes et numéros autour de chaque objet."""
    draw = ImageDraw.Draw(image)
    for obj in objets:
        x1, y1, x2, y2 = obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255), width=3)
        draw.text((x1 + 4, y1 + 2), str(obj['id']), fill=(255, 255, 0))
    return image


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def pipeline(chemin_image, dossier_sortie='resultats'):
    os.makedirs(dossier_sortie, exist_ok=True)

    print("\n" + "=" * 52)
    print("  DÉTECTION ET COMPTAGE D'OBJETS")
    print("=" * 52)

    print("\n[1] Chargement...")
    photo = charger_image(chemin_image)

    print("\n[2] Conversion en niveaux de gris...")
    gris = convertir_gris(photo)
    gris.save(os.path.join(dossier_sortie, '01_niveaux_de_gris.png'))
    print("  [OK]")

    print("\n[3] Lissage (réduction du bruit)...")
    lissee = lisser(gris)
    lissee.save(os.path.join(dossier_sortie, '02_lissage.png'))
    print("  [OK]")

    print("\n[4] Seuillage automatique (Otsu)...")
    t = seuil_otsu(lissee)
    print(f"  Seuil calculé : {t}")
    binaire = binariser(lissee, t)
    binaire.save(os.path.join(dossier_sortie, '03_binaire.png'))
    print("  [OK]")

    print("\n[5] Nettoyage morphologique...")
    propre = nettoyer(binaire)
    propre.save(os.path.join(dossier_sortie, '04_morphologie.png'))
    print("  [OK]")

    print("\n[6] Détection et comptage...")
    img_det, nb, objets = detecter_objets(propre, taille_min=300)
    img_finale = dessiner_boites(img_det, objets)
    img_finale.save(os.path.join(dossier_sortie, '05_detection_finale.png'))

    print("\n" + "=" * 52)
    print(f"  RÉSULTAT : {nb} objet(s) détecté(s)")
    print("=" * 52)
    for obj in objets:
        print(f"  Objet {obj['id']:2d} | "
              f"({obj['x_min']},{obj['y_min']})-({obj['x_max']},{obj['y_max']}) | "
              f"{obj['taille']} pixels")
    print()

    return nb, objets


# ============================================================
# POINT D'ENTRÉE
# ============================================================

if __name__ == '__main__':
    CHEMIN_IMAGE   = 'C:\\Users\\MSI\\OneDrive\\Bureau\\meryem\\FIGL2\\projet_traitement image\\image.png'
    DOSSIER_SORTIE = 'resultats'

    pipeline(CHEMIN_IMAGE, DOSSIER_SORTIE)