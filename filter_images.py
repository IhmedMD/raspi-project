"""
filter_images.py — Parcourt un dossier et identifie les images inadaptées à la détection de feuilles.
Vérifications effectuées sur chaque image :
  • Surexposée      : luminosité moyenne trop élevée OU trop de pixels blancs saturés
  • Sous-exposée    : luminosité moyenne trop faible  OU trop de pixels noirs saturés
  • Faible contraste : écart-type des valeurs de pixels trop faible
  • Floue           : variance du Laplacien (score de netteté) trop faible
  • Sans papier     : la région blanche (papier/disque) couvre moins que MIN_PAPER_COVERAGE
Utilisation :
  python filter_images.py <dossier>          — rapport uniquement (aucun fichier déplacé)
  python filter_images.py <dossier> --move   — déplace les images rejetées vers <dossier>/rejected/
"""

import os
import sys
import shutil

import cv2
import numpy as np

# ── Seuils — à ajuster selon votre configuration ─────────────────────────────
MEAN_OVEREXPOSED   = 210    # valeur moyenne de pixel au-dessus → surexposée
MEAN_UNDEREXPOSED  = 30     # valeur moyenne de pixel en-dessous → sous-exposée
BRIGHT_SAT_FRAC    = 0.35   # fraction de pixels ≥ 250  au-dessus → surexposée
DARK_SAT_FRAC      = 0.35   # fraction de pixels ≤ 5    au-dessus → sous-exposée
MIN_STD            = 20     # écart-type en-dessous → contraste quasi nul
MIN_SHARPNESS      = 50     # variance du Laplacien en-dessous → floue / hors-focus
MIN_PAPER_COVERAGE = 0.03   # fraction de surface papier en-dessous → pas de papier utilisable

EXTENSIONS_SUPPORTEES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# ── Détection du papier (reprend leaf_ml.py / isolate_leaves_high.py) ─────────

def trouver_couverture_papier(gray):
    """
    Retourne la fraction de l'image couverte par de grandes régions claires (disques de papier).
    Utilise la même approche Otsu + morphologie que les scripts de détection.
    """
    h, w = gray.shape
    _, clair = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    noyau = np.ones((15, 15), np.uint8)
    clair = cv2.morphologyEx(clair, cv2.MORPH_CLOSE, noyau, iterations=2)
    clair = cv2.morphologyEx(clair, cv2.MORPH_OPEN,  noyau, iterations=1)

    min_blob = h * w * 0.01
    cnts, _ = cv2.findContours(clair, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    papier = np.zeros((h, w), np.uint8)
    for c in cnts:
        if cv2.contourArea(c) > min_blob:
            cv2.drawContours(papier, [c], -1, 255, -1)

    return float(np.sum(papier > 0)) / (h * w)


# ── Évaluation de la qualité par image ────────────────────────────────────────

def evaluer(chemin):
    """
    Calcule les métriques de qualité pour une image.
    Retourne (raisons: list[str], métriques: dict).
    Une liste de raisons vide signifie que l'image passe tous les contrôles.
    """
    gray = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return ["illisible (format non supporté ou fichier corrompu)"], {}

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    n          = gray.size
    moy        = float(np.mean(gray))
    ecart_type = float(np.std(gray))
    frac_clair = float(np.sum(gray >= 250)) / n
    frac_sombre= float(np.sum(gray <= 5))   / n
    nettete    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    frac_papier= trouver_couverture_papier(gray)

    metriques = {
        "moy":      round(moy,         1),
        "std":      round(ecart_type,  1),
        "clair%":   round(frac_clair  * 100, 1),
        "sombre%":  round(frac_sombre * 100, 1),
        "netteté":  round(nettete,     1),
        "papier%":  round(frac_papier * 100, 1),
    }

    raisons = []

    if moy > MEAN_OVEREXPOSED or frac_clair > BRIGHT_SAT_FRAC:
        raisons.append(
            f"surexposée  (moy={moy:.0f}/255, pixels≥250: {frac_clair*100:.0f}%)"
        )

    if moy < MEAN_UNDEREXPOSED or frac_sombre > DARK_SAT_FRAC:
        raisons.append(
            f"sous-exposée (moy={moy:.0f}/255, pixels≤5: {frac_sombre*100:.0f}%)"
        )

    if ecart_type < MIN_STD:
        raisons.append(f"contraste insuffisant (std={ecart_type:.1f} < {MIN_STD})")

    if nettete < MIN_SHARPNESS:
        raisons.append(f"floue / hors-focus (netteté={nettete:.1f} < {MIN_SHARPNESS})")

    if frac_papier < MIN_PAPER_COVERAGE:
        raisons.append(
            f"pas de papier détecté ({frac_papier*100:.1f}% < {MIN_PAPER_COVERAGE*100:.0f}%)"
        )

    return raisons, metriques


# ── Programme principal ───────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Utilisation :")
        print("  python filter_images.py <dossier>          — rapport uniquement")
        print("  python filter_images.py <dossier> --move   — déplace les rejets dans rejected/")
        return

    dossier   = sys.argv[1]
    deplacer  = "--move" in sys.argv[2:]

    if not os.path.isdir(dossier):
        print(f"❌ Dossier introuvable : {dossier}")
        return

    images = sorted(
        f for f in os.listdir(dossier)
        if os.path.splitext(f)[1].lower() in EXTENSIONS_SUPPORTEES
    )

    if not images:
        print(f"Aucune image trouvée dans '{dossier}'.")
        return

    dossier_rejects = os.path.join(dossier, "rejected")
    if deplacer:
        os.makedirs(dossier_rejects, exist_ok=True)

    print(f"\n📂 Analyse de {len(images)} image(s) dans '{dossier}'")
    print(f"   Mode : {'déplacement des rejets' if deplacer else 'rapport uniquement (--move pour déplacer)'}")
    print("─" * 60)

    liste_ok      = []
    liste_rejetee = []

    for nom in images:
        chemin = os.path.join(dossier, nom)
        raisons, metriques = evaluer(chemin)

        if raisons:
            liste_rejetee.append((nom, raisons, metriques))
        else:
            liste_ok.append((nom, metriques))

    # ── Affichage des images acceptées ───────────────────────────────────────
    print(f"\n✅  Acceptées ({len(liste_ok)}) :")
    if liste_ok:
        for nom, m in liste_ok:
            print(f"   {nom}")
            print(f"      moy={m['moy']}  std={m['std']}  "
                  f"netteté={m['netteté']}  papier={m['papier%']}%")
    else:
        print("   (aucune)")

    # ── Affichage des images rejetées ─────────────────────────────────────────
    print(f"\n✗   Rejetées ({len(liste_rejetee)}) :")
    if liste_rejetee:
        for nom, raisons, metriques in liste_rejetee:
            print(f"   {nom}")
            for r in raisons:
                print(f"      → {r}")
            if deplacer:
                shutil.move(chemin := os.path.join(dossier, nom),
                            os.path.join(dossier_rejects, nom))
                print(f"      → déplacée dans rejected/")
    else:
        print("   (aucune)")

    # ── Résumé ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Total analysé : {len(images)}")
    print(f"  ✅ Acceptées  : {len(liste_ok)}")
    print(f"  ✗  Rejetées   : {len(liste_rejetee)}")
    if deplacer and liste_rejetee:
        print(f"  📁 Dossier rejected : {os.path.abspath(dossier_rejects)}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()