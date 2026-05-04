"""
roi_correct.py — Recadrage et correction de perspective en batch
================================================================
Outil de calibration interactive et de traitement batch pour corriger la perspective

WORKFLOW EN DEUX ÉTAPES :

  Étape 1 — Calibration (une seule fois, sur une image représentative)
  ─────────────────────────────────────────────────────────────────────
  usage: python roi_correct.py --calibrate --image ma_photo.png --config NOM_DU_FICHIER.json

  → Une fenêtre s'ouvre. Clique les 4 coins dans zone d'intérêt dans l'ordre :
      [0] Haut-gauche   [1] Haut-droite
      [2] Bas-droite    [3] Bas-gauche

  → Les paramètres sont sauvegardés dans NOM_DU_FICHIER.json.

  Étape 2 — Traitement batch (Pour le reste des images une fois la calibration faite)
  ─────────────────────────────────────────────────────────
  python roi_correct.py --input ./captures --output ./captures_roi --config NOM_DU_FICHIER.json

  → Chaque image est redressée et recadrée automatiquement.

DÉPENDANCES :
  pip install numpy opencv-python tqdm
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class ROICalibrator:
    """
    classe pour la calibration interactive de la zone d'intérêt (ROI) sur une image de référence. 
    Cliquer sur les 4 coins dans l'ordre : haut-gauche, haut-droite, bas-droite, bas-gauche. Calcule ensuite la matrice d'homographie pour corriger la perspective et recadrer la zone d'intérêt. Sauvegarde la config dans un fichier JSON pour une utilisation ultérieure en batch.
    """

    CORNER_NAMES = [
        "Haut-gauche  [0]",
        "Haut-droite  [1]",
        "Bas-droite   [2]",
        "Bas-gauche   [3]",
    ]
    COLORS = [
        (0, 255, 0),   
        (0, 200, 255), 
        (0, 100, 255),  
        (255, 80, 80),  
    ]
    MAX_DISPLAY = 1200  

    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.original is None:
            raise IOError(f"Impossible de charger l'image : {image_path}")

        H, W = self.original.shape
        self.scale = min(1.0, self.MAX_DISPLAY / W)
        dW, dH = int(W * self.scale), int(H * self.scale)
        self.display = cv2.resize(self.original, (dW, dH))
        # Convertir en BGR pour l'affichage coloré
        self.display_bgr = cv2.cvtColor(self.display, cv2.COLOR_GRAY2BGR)

        self.points_display = []   
        self.points_original = []  

    def _draw_state(self):
        """Redessine l'image avec les points et le guide courant."""
        img = self.display_bgr.copy()
        n = len(self.points_display)

        # Points déjà placés
        for i, (px, py) in enumerate(self.points_display):
            cv2.circle(img, (px, py), 7, self.COLORS[i], -1)
            cv2.circle(img, (px, py), 8, (255, 255, 255), 1)
            cv2.putText(img, str(i), (px + 10, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(img, str(i), (px + 10, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLORS[i], 1)

        # Ligne polygonale entre les points
        if n >= 2:
            for i in range(n - 1):
                cv2.line(img, self.points_display[i], self.points_display[i + 1],
                         (200, 200, 200), 1, cv2.LINE_AA)
        if n == 4:
            cv2.line(img, self.points_display[3], self.points_display[0],
                     (200, 200, 200), 1, cv2.LINE_AA)

        # Instruction courante
        if n < 4:
            msg = f"Clic {n+1}/4 : {self.CORNER_NAMES[n]}"
            color = self.COLORS[n]
        else:
            msg = "4 points places. Appuie sur ENTREE pour confirmer, R pour recommencer."
            color = (180, 255, 180)

        cv2.rectangle(img, (0, 0), (img.shape[1], 32), (30, 30, 30), -1)
        cv2.putText(img, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1,
                    cv2.LINE_AA)

        return img

    def _on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points_display) < 4:
            self.points_display.append((x, y))
            # Remonter aux coordonnées originales
            ox = int(round(x / self.scale))
            oy = int(round(y / self.scale))
            self.points_original.append((ox, oy))
            cv2.imshow(self.win, self._draw_state())

    def run(self) -> list:
        """
        Lance la fenêtre interactive et retourne les 4 coins sélectionnés
        en coordonnées originales (pixels).
        Returns:
            liste de 4 tuples (x, y) dans l'ordre : HG, HD, BD, BG
        """
        self.win = "ROI Calibration — clic 4 coins | ENTREE=valider | R=reset | Q=quitter"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.display_bgr.shape[1], self.display_bgr.shape[0])
        cv2.setMouseCallback(self.win, self._on_click)
        cv2.imshow(self.win, self._draw_state())

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):  # ENTREE
                if len(self.points_display) == 4:
                    break
                else:
                    print(f"  [!] Il faut 4 points ({len(self.points_display)}/4 placés).")
            elif key == ord('r'):  # Reset
                self.points_display.clear()
                self.points_original.clear()
                cv2.imshow(self.win, self._draw_state())
            elif key == ord('q'):
                cv2.destroyAllWindows()
                print("  Calibration annulée.")
                sys.exit(0)

        cv2.destroyAllWindows()
        return self.points_original

#  CALCUL DE LA TRANSFORMATION
# ─────────────────────────────────────────────────────────────────

def compute_transform(src_points: list, output_size: tuple = None) -> dict:
    """
    Calcule la matrice d'homographie et la taille de sortie à partir
    des 4 coins sélectionnés.

    La taille de sortie est déduite automatiquement des distances entre coins
    (moyenne des côtés opposés) pour conserver les proportions réelles.

    Arguments:
        src_points  : liste de 4 (x, y) dans l'image source
        output_size : (width, height) forcé. Si None, calculé automatiquement.

    Returns:
        dict avec 'matrix' (3x3), 'output_width', 'output_height', 'src_points'
    """
    pts = np.array(src_points, dtype=np.float32)
    tl, tr, br, bl = pts[0], pts[1], pts[2], pts[3]

    # Largeur de sortie = moyenne des largeurs haut et bas
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    out_w = int(round((w_top + w_bot) / 2))

    # Hauteur de sortie = moyenne des hauteurs gauche et droite
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    out_h = int(round((h_left + h_right) / 2))

    if output_size is not None:
        out_w, out_h = output_size

    dst = np.array([
        [0,       0      ],
        [out_w-1, 0      ],
        [out_w-1, out_h-1],
        [0,       out_h-1],
    ], dtype=np.float32)

    M, _ = cv2.findHomography(pts, dst)

    return {
        'matrix': M.tolist(),
        'output_width': out_w,
        'output_height': out_h,
        'src_points': [list(p) for p in src_points],
    }


def save_config(config: dict, path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"  Config sauvegardée : {path}")


def load_config(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg['matrix'] = np.array(cfg['matrix'], dtype=np.float64)
    return cfg

#  APPLICATION DE LA TRANSFORMATION
# ─────────────────────────────────────────────────────────────────

def apply_transform(image: np.ndarray, config: dict) -> np.ndarray:
    """
    Applique la correction de perspective et le recadrage définis dans config.

    Args:
        image  : image greyscale 2D uint8
        config : dict chargé depuis le fichier JSON de calibration

    Returns:
        image corrigée uint8, taille (output_height, output_width)
    """
    M = config['matrix']
    out_w = config['output_width']
    out_h = config['output_height']
    return cv2.warpPerspective(image, M, (out_w, out_h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=0)


def apply_clahe(image: np.ndarray, clip: float = 3.0, tile: int = 8) -> np.ndarray:
    """CLAHE local pour rehausser le contraste après correction."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(image)


#  TRAITEMENT BATCH
# ─────────────────────────────────────────────────────────────────

SUPPORTED = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def process_batch(
    input_dir: Path,
    output_dir: Path,
    config: dict,
    use_clahe: bool = False,
    clahe_clip: float = 3.0,
    clahe_tile: int = 8,
    recursive: bool = False,
    preview: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    pattern = '**/*' if recursive else '*'
    files = [f for f in input_dir.glob(pattern)
             if f.is_file() and f.suffix.lower() in SUPPORTED]

    if not files:
        print(f"[!] Aucune image trouvée dans : {input_dir}")
        return {'total': 0, 'success': 0, 'errors': []}

    out_w = config['output_width']
    out_h = config['output_height']
    print(f"\n  Source          : {input_dir}")
    print(f"  Destination     : {output_dir}")
    print(f"  Images          : {len(files)}")
    print(f"  Taille sortie   : {out_w} x {out_h} px")
    print(f"  CLAHE           : {'oui' if use_clahe else 'non'}")
    print()

    success, errors = 0, []

    for img_path in tqdm(files, desc="Correction ROI", unit="img"):
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError("Lecture impossible")

            corrected = apply_transform(img, config)

            if use_clahe:
                corrected = apply_clahe(corrected, clahe_clip, clahe_tile)

            rel = img_path.relative_to(input_dir)
            out_path = (output_dir / rel).with_suffix('.png')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), corrected)

            if preview:
                # Redimensionner l'original à la même hauteur pour le côte-à-côte
                scale = corrected.shape[0] / img.shape[0]
                orig_rs = cv2.resize(img, (int(img.shape[1] * scale), corrected.shape[0]))
                sep = np.full((corrected.shape[0], 4), 200, dtype=np.uint8)
                prev = np.hstack([orig_rs, sep, corrected])
                prev_path = out_path.parent / (out_path.stem + '_preview.png')
                cv2.imwrite(str(prev_path), prev)

            success += 1

        except Exception as e:
            errors.append((img_path.name, str(e)))
            tqdm.write(f"  [ERREUR] {img_path.name} — {e}")

    return {'total': len(files), 'success': success, 'errors': errors}


#  CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Correction de perspective et crop ROI en batch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ÉTAPE 1 — Calibration interactive (une seule fois) :
  python roi_correct.py --calibrate --image photo.png --config roi_config.json

ÉTAPE 2 — Traitement batch :
  python roi_correct.py --input ./captures --output ./captures_roi --config roi_config.json

Avec normalisation CLAHE intégrée :
  python roi_correct.py --input ./captures --output ./out --config roi_config.json --clahe

Avec aperçu côte-à-côte (original | corrigé) :
  python roi_correct.py --input ./captures --output ./out --config roi_config.json --preview
        """
    )

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--calibrate', action='store_true',
                      help="Mode calibration interactive (sélection des 4 coins).")
    mode.add_argument('--input', '-i', type=Path,
                      help="Mode batch : dossier source des images à traiter.")

    # Commun
    parser.add_argument('--config', '-c', type=Path, default=Path('roi_config.json'),
                        help="Fichier de config JSON (sauvegardé en calibration, "
                             "relu en batch). Défaut : roi_config.json")

    # Calibration
    parser.add_argument('--image', type=Path,
                        help="[calibration] Image de référence sur laquelle cliquer les coins.")
    parser.add_argument('--size', type=str, default=None,
                        help="[calibration] Taille de sortie forcée : WIDTHxHEIGHT "
                             "(ex: 800x600). Si absent, déduite des coins cliqués.")

    # Batch
    parser.add_argument('--output', '-o', type=Path,
                        help="[batch] Dossier de destination.")
    parser.add_argument('--clahe', action='store_true',
                        help="[batch] Appliquer CLAHE après la correction.")
    parser.add_argument('--clip', type=float, default=3.0,
                        help="[batch] Paramètre clipLimit CLAHE (défaut : 3.0).")
    parser.add_argument('--tile', type=int, default=8,
                        help="[batch] Taille de tuile CLAHE (défaut : 8).")
    parser.add_argument('--preview', action='store_true',
                        help="[batch] Générer des aperçus original | corrigé.")
    parser.add_argument('--recursive', '-r', action='store_true',
                        help="[batch] Inclure les sous-dossiers.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── MODE CALIBRATION ──────────────────────────────────────────
    if args.calibrate:
        if args.image is None:
            print("[ERREUR] --image est requis en mode --calibrate.")
            sys.exit(1)
        if not args.image.exists():
            print(f"[ERREUR] Image introuvable : {args.image}")
            sys.exit(1)

        print("=" * 60)
        print("  ROI Calibration")
        print("=" * 60)
        print(f"  Image de référence : {args.image}")
        print()
        print("  Dans la fenêtre qui s'ouvre, clique les 4 coins")
        print("  de ta zone d'intérêt dans cet ordre :")
        print()
        print("    [0] Haut-gauche → [1] Haut-droite")
        print("    [2] Bas-droite  → [3] Bas-gauche")
        print()
        print("  Raccourcis : ENTREE = valider | R = reset | Q = quitter")
        print()

        calibrator = ROICalibrator(args.image)
        src_points = calibrator.run()

        print(f"  Points sélectionnés :")
        labels = ["Haut-gauche", "Haut-droite", "Bas-droite ", "Bas-gauche "]
        for label, pt in zip(labels, src_points):
            print(f"    {label} : {pt}")

        output_size = None
        if args.size:
            try:
                w, h = args.size.lower().split('x')
                output_size = (int(w), int(h))
            except Exception:
                print(f"[ERREUR] Format --size invalide : '{args.size}'. Utiliser WIDTHxHEIGHT.")
                sys.exit(1)

        config = compute_transform(src_points, output_size)
        save_config(config, args.config)

        print()
        print(f"  Taille de sortie calculée : {config['output_width']} x {config['output_height']} px")
        print()
        print("  Calibration terminée.")
        print(f"  Lance maintenant le traitement batch avec :")
        print(f"    python roi_correct.py --input ./ton_dossier --output ./sortie --config {args.config}")

    # ── MODE BATCH ────────────────────────────────────────────────
    else:
        if args.output is None:
            print("[ERREUR] --output est requis en mode batch.")
            sys.exit(1)
        if not args.input.exists():
            print(f"[ERREUR] Dossier source introuvable : {args.input}")
            sys.exit(1)
        if not args.config.exists():
            print(f"[ERREUR] Fichier de config introuvable : {args.config}")
            print("  Lance d'abord la calibration :")
            print(f"    python roi_correct.py --calibrate --image photo.png --config {args.config}")
            sys.exit(1)

        print("=" * 60)
        print("  ROI Correct — Traitement batch")
        print("=" * 60)
        print(f"  Config : {args.config}")

        config = load_config(args.config)

        stats = process_batch(
            input_dir=args.input,
            output_dir=args.output,
            config=config,
            use_clahe=args.clahe,
            clahe_clip=args.clip,
            clahe_tile=args.tile,
            recursive=args.recursive,
            preview=args.preview,
        )

        print()
        print("─" * 60)
        print(f"  Résultat : {stats['success']}/{stats['total']} images traitées.")
        if stats['errors']:
            print(f"  Erreurs ({len(stats['errors'])}) :")
            for name, msg in stats['errors']:
                print(f"    • {name} : {msg}")
        print("─" * 60)


if __name__ == '__main__':
    main()