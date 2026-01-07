import librosa
import librosa.display
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure
import os
import warnings
import soundfile as sf
import time

# Configuracion global
warnings.filterwarnings('ignore')

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class AudioForensicsEngine:
    def __init__(self):
        """
        Motor de comparacion de audio basado en Vision Artificial Multi-Escala.
        Metodo: Correlacion Cruzada Normalizada (Template Matching) con invarianza de tempo.
        """
        self.vision_method = cv2.TM_CCOEFF_NORMED

    def load_audio(self, path):
        """Carga el archivo de audio y retorna la serie temporal y la tasa de muestreo."""
        try:
            y, sr = librosa.load(path, sr=22050, duration=None)
            return y, sr
        except Exception as e:
            return None, None

    def generate_fingerprints(self, y, sr):
        """Genera las representaciones espectrales necesarias para el analisis."""
        # 1. Espectrograma de amplitud (STFT)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # 2. Espectrograma Mel (Filtrado 300Hz-8kHz)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=300, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 3. Cromagrama (Notas Musicales) - Crucial para deteccion melodica
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Normalizacion para procesamiento de imagen (0-255)
        img_mel = cv2.normalize(S_dB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_chroma = cv2.normalize(chroma, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Inversion del eje Y para visualizacion estandar
        img_mel = np.flipud(img_mel)
        img_chroma = np.flipud(img_chroma)
        
        return {
            'y': y, 'sr': sr,
            'stft': D, 'mel_db': S_dB, 'chroma': chroma,
            'img_mel': img_mel, 'img_chroma': img_chroma
        }

    def scan_for_plagiarism(self, data_orig, data_susp):
        """
        Ejecuta un barrido multi-escala buscando el patron original dentro del sospechoso.
        Retorna el score de coincidencia y la ubicacion temporal.
        """
        img_needle = data_orig['img_chroma']
        img_haystack = data_susp['img_chroma']
        
        h_n, w_n = img_needle.shape
        h_h, w_h = img_haystack.shape

        # Definir ventana de busqueda (max 1300 px aprox 30-40s)
        search_window_px = min(w_n, 1300) 
        needle_crop = img_needle[:, 0:search_window_px]

        best_score = -1
        best_scale = 1.0
        best_loc = 0
        best_width_scaled = 0

        # Rango de escalas de tiempo (0.8x a 1.2x) para detectar cambios de tempo
        scales = np.linspace(0.8, 1.2, 15)

        for scale in scales:
            new_w = int(needle_crop.shape[1] * scale)
            if new_w > w_h: continue
            
            needle_resized = cv2.resize(needle_crop, (new_w, h_n))
            
            # Ejecutar correlacion cruzada
            res = cv2.matchTemplate(img_haystack, needle_resized, self.vision_method)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_scale = scale
                best_loc = max_loc[0]
                best_width_scaled = new_w

        # Calculo de score final normalizado
        final_score = best_score * 100
        
        # Verificacion secundaria con HOG (Textura)
        texture_score = 0
        if best_width_scaled > 0:
            roi = img_haystack[:, best_loc : best_loc + best_width_scaled]
            if roi.shape[1] > 0:
                roi_resized = cv2.resize(roi, (128, 64))
                needle_hog_resized = cv2.resize(needle_crop, (128, 64))
                
                fd1, _ = hog(needle_hog_resized, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
                fd2, _ = hog(roi_resized, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
                
                norm_fd1 = np.linalg.norm(fd1)
                norm_fd2 = np.linalg.norm(fd2)
                if norm_fd1 > 0 and norm_fd2 > 0:
                    texture_score = np.dot(fd1, fd2) / (norm_fd1 * norm_fd2)

        # Score ponderado (70% Visual, 30% Textura)
        confidence = (final_score * 0.7) + (texture_score * 100 * 0.3)

        # Calculo de tiempo de inicio (Samples -> Segundos)
        # Hop length default de librosa es 512
        time_start_susp = (best_loc * 512) / data_susp['sr']
        
        return {
            'score': confidence,
            'raw_visual': best_score,
            'is_plagiarism': confidence > 45.0, # Umbral de decision
            'scale': best_scale,
            'time_start': time_start_susp,
            'match_rect': (best_loc, best_width_scaled),
            'img_needle': needle_crop,
            'img_haystack': img_haystack
        }

    def visualize_full_comparison(self, d1, name1, d2, name2):
        """Genera panel de control con 4 visualizaciones comparativas."""
        print(f"[INFO] Generando graficas de analisis espectral...")
        fig, ax = plt.subplots(4, 2, figsize=(18, 12))
        fig.suptitle(f'ANALISIS ESPECTRAL COMPARATIVO\nReferencia: {name1}  |  Objetivo: {name2}', fontsize=14, fontweight='bold')

        # 1. Forma de Onda
        librosa.display.waveshow(d1['y'], sr=d1['sr'], ax=ax[0,0], color='#1f77b4', alpha=0.8); ax[0,0].set_title(f"Onda ({name1})")
        librosa.display.waveshow(d2['y'], sr=d2['sr'], ax=ax[0,1], color='#ff7f0e', alpha=0.8); ax[0,1].set_title(f"Onda ({name2})")

        # 2. STFT
        librosa.display.specshow(d1['stft'], y_axis='log', x_axis='time', ax=ax[1,0], cmap='magma'); ax[1,0].set_title("Espectrograma STFT")
        librosa.display.specshow(d2['stft'], y_axis='log', x_axis='time', ax=ax[1,1], cmap='magma'); ax[1,1].set_title("Espectrograma STFT")

        # 3. Mel
        librosa.display.specshow(d1['mel_db'], y_axis='mel', x_axis='time', ax=ax[2,0], cmap='inferno'); ax[2,0].set_title("Espectrograma Mel (Filtrado)")
        librosa.display.specshow(d2['mel_db'], y_axis='mel', x_axis='time', ax=ax[2,1], cmap='inferno'); ax[2,1].set_title("Espectrograma Mel (Filtrado)")

        # 4. Cromagrama
        librosa.display.specshow(d1['chroma'], y_axis='chroma', x_axis='time', ax=ax[3,0], cmap='coolwarm'); ax[3,0].set_title("Cromagrama (Notas)")
        librosa.display.specshow(d2['chroma'], y_axis='chroma', x_axis='time', ax=ax[3,1], cmap='coolwarm'); ax[3,1].set_title("Cromagrama (Notas)")

        plt.tight_layout()
        print("[INFO] Visualizacion generada. Cierre la ventana para continuar.")
        plt.show()

    def visualize_evidence_report(self, res, name_susp):
        """Genera el reporte final con localizacion del hallazgo."""
        print(f"[INFO] Generando reporte forense...")
        fig = plt.figure(figsize=(16, 10))
        
        # Patron buscado
        ax1 = plt.subplot(3, 1, 1)
        plt.imshow(res['img_needle'], cmap='jet', aspect='auto')
        plt.title("PATRON DE REFERENCIA (Extracto Original)", fontweight='bold')
        plt.axis('off')

        # Mapa de busqueda
        ax2 = plt.subplot(3, 1, 2)
        plt.imshow(res['img_haystack'], cmap='jet', aspect='auto')
        plt.title(f"MAPA DE ANALISIS: {name_susp}", fontweight='bold')
        
        # Marcador de hallazgo
        x, w = res['match_rect']
        rect = plt.Rectangle((x, 0), w, res['img_haystack'].shape[0], linewidth=3, edgecolor='#00ff00', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x, 10, f"COINCIDENCIA DETECTADA ({res['score']:.1f}%)", color='#00ff00', fontweight='bold')
        plt.axis('off')

        # Datos del reporte
        ax3 = plt.subplot(3, 1, 3); ax3.axis('off')
        
        status = "POSITIVO (COINCIDENCIA ENCONTRADA)" if res['is_plagiarism'] else "NEGATIVO (NO CONCLUYENTE)"
        time_str = time.strftime('%M:%S', time.gmtime(res['time_start']))
        
        report_text = (
            f"REPORTE DE ANALISIS FORENSE DE AUDIO\n"
            f"====================================\n"
            f"ESTADO: {status}\n\n"
            f"METRICAS:\n"
            f" - Correlacion Visual: {res['raw_visual']*100:.2f}%\n"
            f" - Escala Temporal:    {res['scale']:.2f}x (Modificacion de Tempo)\n"
            f" - Ubicacion Temporal: {time_str} (Min:Seg)\n"
            f" - CONFIDENCE SCORE:   {res['score']:.2f}%\n"
        )
        
        props = dict(boxstyle='round', facecolor='#f0f0f0', alpha=1.0, edgecolor='black')
        ax3.text(0.5, 0.5, report_text, ha='center', va='center', fontsize=11, bbox=props, family='monospace')

        plt.tight_layout()
        print("[INFO] Reporte generado. Cierre la ventana para reproduccion de evidencia.")
        plt.show()

# ==========================================
# FUNCIONES DE SISTEMA
# ==========================================

def get_file_list():
    if not os.path.exists("./audios"): os.makedirs("./audios")
    return sorted([f for f in os.listdir("./audios") if f.lower().endswith(('.mp3', '.wav', '.flac'))])

def play_exact_evidence(res, data_orig, data_susp, name1, name2):
    if not HAS_WINSOUND: return
    print("\n[AUDIO] Reproduciendo evidencia (Clip 10s)...")
    
    sr = data_susp['sr']
    start_sec = res['time_start']
    
    # Recorte de 10 segundos
    start_sample = int(start_sec * sr)
    end_sample = start_sample + (10 * sr)
    
    if end_sample > len(data_susp['y']): end_sample = len(data_susp['y'])
    
    clip_plagio = data_susp['y'][start_sample:end_sample]
    clip_orig = data_orig['y'][:10*sr] # Referencia
    
    # Reproduccion secuencial
    print(f" -> Reproduciendo Referencia: {name1}")
    sf.write("TEMP_REF.wav", clip_orig, sr)
    winsound.PlaySound("TEMP_REF.wav", winsound.SND_FILENAME)
    time.sleep(0.5)
    
    print(f" -> Reproduciendo Hallazgo: {name2} (@ {int(start_sec//60)}:{int(start_sec%60):02d})")
    sf.write("TEMP_BAD.wav", clip_plagio, sr)
    winsound.PlaySound("TEMP_BAD.wav", winsound.SND_FILENAME)
    
    try:
        os.remove("TEMP_REF.wav")
        os.remove("TEMP_BAD.wav")
    except: pass

def main():
    print("==========================================================")
    print("   SISTEMA DE ANALISIS  DE AUDIO")
    print("==========================================================")
    
    engine = AudioForensicsEngine()
    
    while True:
        print("\nOpciones:")
        print("1. Analisis de Base de Datos")
        print("2. Salir")
        
        try:
            op = input("Seleccion: ")
        except KeyboardInterrupt:
            break
        
        if op == '1':
            files = get_file_list()
            if not files: 
                print("[ERROR] No se encontraron archivos en la carpeta ./audios")
                continue
            
            print("\n--- SELECCION DE ARCHIVO DE REFERENCIA (ORIGINAL) ---")
            for i, f in enumerate(files): print(f"{i+1}. {f}")
            
            try:
                idx = int(input("Ingrese numero de archivo: ")) - 1
                if idx < 0 or idx >= len(files): raise ValueError
                
                target_path = f"./audios/{files[idx]}"
                target_name = files[idx]
            except:
                print("[ERROR] Seleccion invalida.")
                continue
            
            print(f"\n[PROCESO] Cargando referencia: {target_name}...")
            data_orig = engine.generate_fingerprints(*engine.load_audio(target_path))
            if data_orig['y'] is None: 
                print("[ERROR] Fallo al cargar audio.")
                continue

            others = [f for f in files if f != target_name]
            print(f"\n[PROCESO] Iniciando escaneo de {len(others)} archivos...")
            print("-" * 75)
            print(f"{'ARCHIVO':<45} | {'SCORE':<8} | {'ESTADO'}")
            print("-" * 75)
            
            best_score = -1
            best_res = None
            best_name = ""
            best_data_susp = None
            
            # Barrido de base de datos
            for f in others:
                path = f"./audios/{f}"
                data_susp = engine.generate_fingerprints(*engine.load_audio(path))
                
                if data_susp['y'] is None: continue
                
                res = engine.scan_for_plagiarism(data_orig, data_susp)
                score = res['score']
                
                status = "ALERTA" if res['is_plagiarism'] else "OK"
                print(f"{f[:43]:<45} | {score:>5.1f}%  | {status}")
                
                if score > best_score:
                    best_score = score
                    best_res = res
                    best_name = f
                    best_data_susp = data_susp
            
            print("-" * 75)
            
            # Resultado del analisis
            if best_res and best_score > 30:
                print(f"\n[RESULTADO] Mayor coincidencia detectada: {best_name}")
                print(f"            Score de Confianza: {best_score:.1f}%")
                
                if best_res['is_plagiarism']:
                    print("[ACCION] Iniciando protocolo de visualizacion y evidencia...")
                    
                    # 1. Visualizacion Comparativa
                    engine.visualize_full_comparison(data_orig, target_name, best_data_susp, best_name)
                    
                    # 2. Reporte Forense
                    engine.visualize_evidence_report(best_res, best_name)
                    
                    # 3. Reproduccion
                    play_exact_evidence(best_res, data_orig, best_data_susp, target_name, best_name)
                else:
                    print("[RESULTADO] El score mas alto no supera el umbral de plagio definido.")
            else:
                print("[RESULTADO] No se encontraron coincidencias significativas.")

        elif op == '2':
            break

if __name__ == "__main__":
    main()