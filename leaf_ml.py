"""
Algorithme de segmentation ML pour feuilles sur fond blanc
================================================

Pipeline:
Usage: python leaf_ml.py <chemin_image> [chemin_sortie]

Interactive display: A=refaire tourner le code, 2/3=changer le nombre de clusters, S=sauvegarder, Q=quitter (sans sauvegarder). Clic droit sur une région pour la supprimer.

Dependencies: opencv-python, scikit-learn, scikit-image, numpy
Install:  pip install opencv-python scikit-learn scikit-image numpy
"""

import os
import sys
import time

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Tunable config ────────────────────────────────────────────────────────────
MAX_DIM    = 1600   # longest side used for feature extraction — lower = faster on Pi
N_CLUSTERS = 2     # 2 = leaf vs background; press 3 in window to try 3 clusters
MIN_AREA   = 100   # minimum contour area (px²) to keep as a leaf region

# ── Global state ──────────────────────────────────────────────────────────────
gray               = None
outlined           = None
leaf_mask          = None
win_scale          = 1.0
detected_contours  = []   # kept in sync so right-click can remove individual regions


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(gray_img):
    """Light bilateral + CLAHE — contrast is already good with white paper background."""
    bilateral = cv2.bilateralFilter(gray_img, 9, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    return clahe.apply(bilateral)


def find_paper_mask(gray_img):
    """
    Detect bright paper/disc regions (the white background leaves sit on).
    Uses Otsu to find the brightest pixels, then keeps only large blobs
    so small bright noise is discarded.
    """
    h, w = gray_img.shape
    _, bright = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((15, 15), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=3)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  kernel, iterations=2)

    # Keep only regions ≥1% of image (actual discs, not noise)
    min_paper = (h * w) * 0.01
    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    paper_mask = np.zeros((h, w), np.uint8)
    for c in cnts:
        if cv2.contourArea(c) > min_paper:
            cv2.drawContours(paper_mask, [c], -1, 255, -1)

    cv2.imwrite("debug_paper_mask.png", paper_mask)
    return paper_mask


# ── Feature extraction ────────────────────────────────────────────────────────

def _gabor(img_f32, frequency, theta):
    """OpenCV Gabor filter — faster than scikit-image on ARM/Pi."""
    lam = 1.0 / frequency
    kernel = cv2.getGaborKernel((21, 21), 3.0, theta, lam, 0.5, 0, ktype=cv2.CV_32F)
    s = kernel.sum()
    if s != 0:
        kernel /= s
    return cv2.filter2D(img_f32, cv2.CV_32F, kernel)


def extract_features(gray_img):
    """
    Build an (h*w, n_features) float32 matrix tuned for low-contrast images.

    Features:
      0-3  : Gaussian-blurred intensity at 4 scales      (neighbourhood context)
      4-6  : Difference of Gaussians at 3 scale pairs    (amplifies subtle intensity differences)
      7    : Local contrast  (pixel / local mean)         (normalises illumination variation)
      8    : Sobel gradient magnitude                     (leaf edges)
      9    : Laplacian of Gaussian                        (blob/region detection)
      10-13: Gabor at 4 orientations, 2 frequencies       (leaf surface texture)
      14   : Local Binary Pattern                         (micro-texture)
    """
    img_f = gray_img.astype(np.float32) / 255.0
    feats = []

    # Multi-scale intensity
    blurs = {}
    for sigma in (0, 2, 4, 8):
        b = img_f if sigma == 0 else cv2.GaussianBlur(img_f, (0, 0), float(sigma))
        blurs[sigma] = b
        feats.append(b.ravel())

    # Difference of Gaussians — bandpass filter that amplifies subtle differences
    # between scales; very effective when absolute contrast is low
    for s1, s2 in ((0, 2), (2, 4), (4, 8)):
        feats.append((blurs[s1] - blurs[s2]).ravel())

    # Local contrast: how different is each pixel from its neighbourhood mean?
    # Normalises out slow illumination gradients common in low-contrast images
    local_mean = cv2.GaussianBlur(img_f, (0, 0), 8.0)
    local_contrast = img_f - local_mean
    feats.append(local_contrast.ravel())

    # Gradient magnitude
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    feats.append(np.sqrt(gx ** 2 + gy ** 2).ravel())

    # Laplacian of Gaussian
    smoothed = cv2.GaussianBlur(img_f, (0, 0), 2.0)
    feats.append(cv2.Laplacian(smoothed, cv2.CV_32F).ravel())

    # Gabor texture: 4 orientations × 2 frequencies
    # More orientations capture anisotropic leaf venation patterns
    for freq in (0.10, 0.20):
        for theta in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
            feats.append(_gabor(img_f, frequency=freq, theta=theta).ravel())

    # Local Binary Pattern (micro-texture)
    lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform").astype(np.float32)
    feats.append(lbp.ravel())

    return np.column_stack(feats)  # (h*w, 15)


# ── ML segmentation ───────────────────────────────────────────────────────────

def _downsample(img, max_dim):
    """Resize image so its longest side ≤ max_dim. Returns (small_img, factor)."""
    h, w = img.shape[:2]
    factor = min(1.0, max_dim / max(h, w))
    if factor < 1.0:
        small = cv2.resize(img, (int(w * factor), int(h * factor)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img.copy()
    return small, factor


def _cleanup_and_filter(mask, orig_h, orig_w, factor, min_area, max_frac=0.10):
    """Upsample mask, morphological cleanup, filter contours by area."""
    if factor < 1.0:
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = orig_h * orig_w * max_frac
    cnts = [c for c in cnts if min_area < cv2.contourArea(c) < max_area]
    final = np.zeros((orig_h, orig_w), np.uint8)
    for c in cnts:
        cv2.drawContours(final, [c], -1, 255, -1)
    return final, cnts


def segment_within_paper(gray_img, paper_mask, n_clusters, min_area):
    """
    ML pipeline restricted to paper pixels only.
    K-means runs solely on pixels inside the paper mask, then the leaf cluster
    is identified as the one with the LOWEST mean raw intensity — leaves are
    darker than the white paper, making this a reliable and unambiguous signal.
    """
    orig_h, orig_w = gray_img.shape
    small, factor = _downsample(gray_img, MAX_DIM)
    small_paper, _ = _downsample(paper_mask, MAX_DIM)
    sh, sw = small.shape

    print(f"   Extraction features sur {sw}×{sh}px (mode papier)...")
    t0 = time.time()
    X_all = extract_features(small)
    print(f"   {X_all.shape[1]} features × {X_all.shape[0]} pixels  ({time.time()-t0:.1f}s)")

    # Select only paper pixels
    paper_flat = small_paper.ravel() > 0
    X_paper = X_all[paper_flat]
    gray_paper = small.ravel()[paper_flat].astype(np.float32)

    if X_paper.shape[0] < n_clusters * 10:
        print("   ⚠️  Pas assez de pixels papier.")
        return np.zeros((orig_h, orig_w), np.uint8), []

    X_scaled = StandardScaler().fit_transform(X_paper)

    print(f"   K-means (k={n_clusters}) sur {X_paper.shape[0]} pixels papier...")
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    paper_labels = km.fit_predict(X_scaled)
    print(f"   Clustering terminé  ({time.time()-t0:.1f}s)")

    # Rank clusters by mean raw intensity — darkest = leaves (paper is bright)
    intensities = [(np.mean(gray_paper[paper_labels == lbl]), lbl)
                   for lbl in range(n_clusters)]
    intensities.sort()
    print(f"   Intensités: {[(f'{v:.1f}', l) for v, l in intensities]}")

    # Exclude the brightest cluster (paper itself); keep the rest as leaves
    bg_label = intensities[-1][1]
    leaf_set  = {lbl for _, lbl in intensities[:-1]}
    print(f"   Cluster papier: {bg_label}  |  Clusters feuilles: {leaf_set}")

    # Reconstruct mask on the small image
    small_mask_flat = np.zeros(sh * sw, np.uint8)
    paper_indices = np.where(paper_flat)[0]
    for i, lbl in enumerate(paper_labels):
        if lbl in leaf_set:
            small_mask_flat[paper_indices[i]] = 255
    small_mask = small_mask_flat.reshape(sh, sw)

    return _cleanup_and_filter(small_mask, orig_h, orig_w, factor, min_area)


def segment_full_image(gray_img, n_clusters, min_area):
    """
    Generic ML pipeline for images without a clear paper background.
    Clusters the full image; identifies background by border-touching heuristic.
    """
    orig_h, orig_w = gray_img.shape
    small, factor = _downsample(gray_img, MAX_DIM)
    sh, sw = small.shape

    print(f"   Extraction features sur {sw}×{sh}px (mode générique)...")
    t0 = time.time()
    X = extract_features(small)
    print(f"   {X.shape[1]} features × {X.shape[0]} pixels  ({time.time()-t0:.1f}s)")

    X_scaled = StandardScaler().fit_transform(X)

    print(f"   K-means (k={n_clusters}, n_init=5)...")
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0)
    labels = km.fit_predict(X_scaled).reshape(sh, sw).astype(np.uint8)
    print(f"   Clustering terminé  ({time.time()-t0:.1f}s)")

    # Background = cluster with most border pixels
    margin = max(3, sh // 50)
    bm = np.zeros((sh, sw), bool)
    bm[:margin, :] = bm[-margin:, :] = bm[:, :margin] = bm[:, -margin:] = True
    scores = [(int(np.sum((labels == lbl) & bm)), lbl) for lbl in range(n_clusters)]
    scores.sort(reverse=True)
    bg_label   = scores[0][1]
    leaf_labels = {lbl for lbl in range(n_clusters) if lbl != bg_label}
    print(f"   Cluster fond: {bg_label}  |  Clusters feuilles: {leaf_labels}")

    small_mask = np.zeros((sh, sw), np.uint8)
    for lbl in leaf_labels:
        small_mask[labels == lbl] = 255

    mask, cnts = _cleanup_and_filter(small_mask, orig_h, orig_w, factor, min_area,
                                     max_frac=0.50)
    coverage = np.sum(mask > 0) / mask.size
    if coverage > 0.70:
        print(f"   ⚠️  Couverture {coverage*100:.0f}% — clustering ambigu, essaie k=3 (touche 3).")
        return np.zeros((orig_h, orig_w), np.uint8), []
    return mask, cnts


def segment_leaves(gray_img, n_clusters=2, min_area=200):
    """
    Orchestrator: detects white paper first; if found, runs the focused
    paper-pixel ML pipeline. Otherwise falls back to full-image clustering.
    """
    paper_mask = find_paper_mask(gray_img)
    paper_coverage = np.sum(paper_mask > 0) / paper_mask.size

    if paper_coverage > 0.02:
        print(f"   📄 Papier détecté ({paper_coverage*100:.1f}%) — clustering restreint au papier.")
        mask, cnts = segment_within_paper(gray_img, paper_mask, n_clusters, min_area)
        if cnts:
            return mask, cnts
        print("   ⚠️  Aucune feuille dans le papier — mode générique.")

    print("   ↩️  Mode générique (pleine image).")
    return segment_full_image(gray_img, n_clusters, min_area)


# ── Detection + display helpers ───────────────────────────────────────────────

def _redraw():
    """Redraw outlined and leaf_mask from the current detected_contours list."""
    global outlined, leaf_mask
    outlined  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    leaf_mask = np.zeros_like(gray)
    for c in detected_contours:
        cv2.drawContours(outlined,  [c], -1, (0, 255, 0), 2)
        cv2.drawContours(leaf_mask, [c], -1, 255, -1)


def remove_contour_at(click_x, click_y):
    """Remove the first contour whose interior contains (click_x, click_y)."""
    for i, c in enumerate(detected_contours):
        if cv2.pointPolygonTest(c, (float(click_x), float(click_y)), False) >= 0:
            detected_contours.pop(i)
            _redraw()
            print(f"🗑️  Région supprimée. {len(detected_contours)} restante(s).")
            return
    print("   Aucune région détectée à cet endroit.")


def run_detection(n_clusters):
    global outlined, leaf_mask, detected_contours
    outlined  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    leaf_mask = np.zeros_like(gray)

    print(f"\n🔍 Segmentation ML (k={n_clusters})...")
    mask, contours = segment_leaves(preprocess(gray),
                                    n_clusters=n_clusters,
                                    min_area=MIN_AREA)
    detected_contours = list(contours)
    leaf_mask = mask
    for c in contours:
        cv2.drawContours(outlined, [c], -1, (0, 255, 0), 2)
    print(f"✅ {len(contours)} région(s) détectée(s).")


def save_results(output_path):
    cv2.imwrite(output_path, outlined)

    mask_path = output_path.replace(".png", "_mask.png")
    cv2.imwrite(mask_path, leaf_mask)

    extracted = cv2.bitwise_and(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        mask=leaf_mask,
    )
    extracted_path = output_path.replace(".png", "_extracted.png")
    cv2.imwrite(extracted_path, extracted)

    print(f"💾 Contours  : {os.path.abspath(output_path)}")
    print(f"💾 Masque    : {os.path.abspath(mask_path)}")
    print(f"💾 Extraites : {os.path.abspath(extracted_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(image_path, output_path="plants_ml_outline.png"):
    global gray, outlined, leaf_mask, win_scale

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"❌ Impossible de lire '{image_path}'")
        return

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = gray.shape
    win_scale = min(600 / w, 900 / h, 1.0)

    outlined  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    leaf_mask = np.zeros_like(gray)
    n_clusters = N_CLUSTERS

    win_name = "ML | Clic D=supprimer | A=relancer | 2/3=clusters | S=sauver | Q=quitter"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, int(w * win_scale), int(h * win_scale))

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_RBUTTONDOWN:
            remove_contour_at(int(mx / win_scale), int(my / win_scale))
            cv2.imshow(win_name, cv2.resize(outlined, (int(w * win_scale), int(h * win_scale))))

    cv2.setMouseCallback(win_name, on_mouse)

    run_detection(n_clusters)
    cv2.imshow(win_name, cv2.resize(outlined, (int(w * win_scale), int(h * win_scale))))

    print("=" * 55)
    print("🖱️  CLIC DROIT → supprimer une région détectée")
    print("⌨️  A          → relancer la détection")
    print("⌨️  2 / 3      → changer le nombre de clusters")
    print("⌨️  S          → sauvegarder les résultats")
    print("⌨️  Q/ESC      → quitter")
    print("=" * 55)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("a"), ord("A")):
            run_detection(n_clusters)
            cv2.imshow(win_name, cv2.resize(outlined, (int(w * win_scale), int(h * win_scale))))

        elif key == ord("2") and n_clusters != 2:
            n_clusters = 2
            print("→ Clusters: 2")
            run_detection(n_clusters)
            cv2.imshow(win_name, cv2.resize(outlined, (int(w * win_scale), int(h * win_scale))))

        elif key == ord("3") and n_clusters != 3:
            n_clusters = 3
            print("→ Clusters: 3")
            run_detection(n_clusters)
            cv2.imshow(win_name, cv2.resize(outlined, (int(w * win_scale), int(h * win_scale))))

        elif key in (ord("s"), ord("S")):
            save_results(output_path)

        elif key in (ord("q"), 27):
            break

        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("👋 Terminé.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python leaf_ml.py <image_path> [output_path]")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else "plants_ml_outline.png"
        main(sys.argv[1], out)
