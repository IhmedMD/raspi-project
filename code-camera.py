#!/usr/bin/env python3
"""
Script de capture automatique pour caméra Vimba X
Lance la capture en timelapse dès le démarrage
Ne pas oublier d'avoir Vimba X installé et la caméra connectée avant de lancer le script.
Aussi d'installer les librairies nécessaires (vmbpy, opencv-python).
Dépendances : 
    vmbpy opencv-python
"""

from vmbpy import *
import cv2
import time
from datetime import datetime
import os
import sys

# ===== CONFIGURATION =====
# Modifie ces valeurs selon ce qui est voulu
INTERVAL_SECONDS = 5        # Intervalle entre chaque photo (en secondes)
TOTAL_IMAGES = 1           # Nombre total de photos (-1 = infini)
FOLDER = "captures"         # Dossier de sauvegarde
EXPOSURE_AUTO = False        # True = exposition auto, False = manuel
EXPOSURE_TIME = 10000       # Temps d'exposition en µs (si EXPOSURE_AUTO = False)
GAIN_AUTO = False            # True = gain auto, False = manuel
GAIN = 0                    # Gain en dB (si GAIN_AUTO = False)
# =========================

def main():
    print("=" * 50)
    print("CAPTURE AUTOMATIQUE - CAMÉRA VIMBA X")
    print("=" * 50)
    
    # Créer le dossier s'il n'existe pas
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)
        print(f"✓ Dossier créé: {FOLDER}/")
    
    print(f"\n📷 Configuration:")
    print(f"   - Intervalle: {INTERVAL_SECONDS} secondes")
    if TOTAL_IMAGES == -1:
        print(f"   - Mode: INFINI (Ctrl+C pour arrêter)")
    else:
        print(f"   - Nombre d'images: {TOTAL_IMAGES}")
        print(f"   - Durée estimée: {(TOTAL_IMAGES * INTERVAL_SECONDS) / 60:.1f} minutes")
    print(f"   - Dossier: {FOLDER}/")
    print(f"   - Exposition: {'AUTO' if EXPOSURE_AUTO else f'MANUEL ({EXPOSURE_TIME} µs)'}")
    print(f"   - Gain: {'AUTO' if GAIN_AUTO else f'MANUEL ({GAIN} dB)'}")
    
    try:
        # Initialiser Vimba
        with VmbSystem.get_instance() as vmb:
            # Détecter les caméras
            cameras = vmb.get_all_cameras()
            
            if not cameras:
                print("\n❌ ERREUR: Aucune caméra détectée!")
                print("   Vérifiez que la caméra est branchée et allumée.")
                sys.exit(1)
            
            print(f"\n✓ Caméra détectée: {cameras[0].get_name()}")
            
            # Connecter à la première caméra
            with cameras[0] as cam:
                print(f"✓ Connexion établie")
                
                # Configuration
                try:
                    # Exposition
                    if EXPOSURE_AUTO:
                        cam.ExposureAuto.set('Continuous')
                        print(f"✓ Exposition automatique activée")
                    else:
                        cam.ExposureAuto.set('Off')
                        cam.ExposureTime.set(EXPOSURE_TIME)
                        print(f"✓ Exposition manuelle: {EXPOSURE_TIME} µs")
                    
                    # Gain
                    if GAIN_AUTO:
                        cam.GainAuto.set('Continuous')
                        print(f"✓ Gain automatique activé")
                    else:
                        cam.GainAuto.set('Off')
                        cam.Gain.set(GAIN)
                        print(f"✓ Gain manuel: {GAIN} dB")
                    
                    # Format pixel
                    try:
                        cam.PixelFormat.set('BGR8')
                        print(f"✓ Format: BGR8 (couleur)")
                    except:
                        cam.PixelFormat.set('Mono8')
                        print(f"✓ Format: Mono8 (noir et blanc)")
                    
                except Exception as e:
                    print(f"⚠️  Avertissement configuration: {e}")
                
                # Informations caméra
                try:
                    width = cam.Width.get()
                    height = cam.Height.get()
                    print(f"✓ Résolution: {width}x{height}")
                except:
                    pass
                
                print("\n" + "=" * 50)
                print("DÉBUT DE LA CAPTURE")
                print("=" * 50)
                print("Appuyez sur Ctrl+C pour arrêter\n")
                
                # Démarrer la capture
                image_count = 0
                start_time = time.time()
                
                while True:
                    # Vérifier si on a atteint le nombre d'images
                    if TOTAL_IMAGES != -1 and image_count >= TOTAL_IMAGES:
                        print(f"\n✓ Objectif atteint: {TOTAL_IMAGES} images")
                        break
                    
                    # Timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{FOLDER}/img_{timestamp}_{image_count:05d}.jpg"
                    
                    # Capturer
                    try:
                        frame = cam.get_frame()
                        image = frame.as_opencv_image()
                        cv2.imwrite(filename, image)
                        
                        image_count += 1
                        elapsed_minutes = (time.time() - start_time) / 60
                        
                        # Affichage
                        if TOTAL_IMAGES == -1:
                            print(f"[{image_count}] {timestamp} → {filename} ({elapsed_minutes:.1f} min)")
                        else:
                            remaining = TOTAL_IMAGES - image_count
                            print(f"[{image_count}/{TOTAL_IMAGES}] {timestamp} → {filename} (restant: {remaining})")
                        
                    except Exception as e:
                        print(f"❌ Erreur capture #{image_count}: {e}")
                        continue
                    
                    # Attendre avant la prochaine capture
                    if TOTAL_IMAGES == -1 or image_count < TOTAL_IMAGES:
                        time.sleep(INTERVAL_SECONDS)
                
                # Résumé final
                total_time = (time.time() - start_time) / 60
                print("\n" + "=" * 50)
                print("CAPTURE TERMINÉE")
                print("=" * 50)
                print(f"✓ Images capturées: {image_count}")
                print(f"✓ Durée totale: {total_time:.2f} minutes")
                print(f"✓ Images dans: {FOLDER}/")
                
    except KeyboardInterrupt:
        # Arrêt par Ctrl+C
        total_time = (time.time() - start_time) / 60
        print("\n\n" + "=" * 50)
        print("CAPTURE ARRÊTÉE (Ctrl+C)")
        print("=" * 50)
        print(f"✓ Images capturées: {image_count}")
        print(f"✓ Durée totale: {total_time:.2f} minutes")
        print(f"✓ Images dans: {FOLDER}/")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()