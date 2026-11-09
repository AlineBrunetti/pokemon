"""
Passo 3 - Reconhece objetos em um video (ou na webcam).
Melhorias aplicadas:
1 - Suavizacao e limiar de confianca (utilizando collections.deque)
2 - Linha do tempo de aparicoes do pokemon
"""
import sys
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from utils import letterbox

CLASSE_VAZIA = "fundo"
LIMIAR_CONFIANCA = 0.60
JANELA_FRAMES = 8 # Avalia ultimos 8 frames para media

model = tf.keras.models.load_model("modelo.keras")
class_names = json.load(open("class_names.json"))

fonte = sys.argv[1] if len(sys.argv) > 1 else "0"
webcam = fonte.isdigit()
cap = cv2.VideoCapture(int(fonte) if webcam else fonte)

if not cap.isOpened():
    raise SystemExit(f"Nao consegui abrir: {fonte}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
writer = None
if not webcam:
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    saida = f"saida_{os.path.splitext(os.path.basename(fonte))[0]}.mp4"
    writer = cv2.VideoWriter(saida, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

n = 0
# Variavel para suavizacao
historico_probs = deque(maxlen=JANELA_FRAMES)
# Variavel para linha do tempo
historico_timeline = []

while True:
    ok, frame = cap.read()
    if not ok:
        break
    if webcam:
        frame = cv2.flip(frame, 1)
        
    img = cv2.cvtColor(letterbox(frame), cv2.COLOR_BGR2RGB).astype("float32")
    probs = model.predict(img[None], verbose=0)[0]
    
    # MELHORIA: Suavizacao e Limiar de Confianca
    historico_probs.append(probs)
    probs_media = np.mean(historico_probs, axis=0)
    
    i = int(np.argmax(probs_media))
    if probs_media[i] < LIMIAR_CONFIANCA:
        rotulo = "?"
    else:
        rotulo = class_names[i]
        
    # Adicionando na lista da timeline para o relatorio do fim
    historico_timeline.append(rotulo)
    
    # --- desenho ---
    texto = "nenhum objeto" if rotulo == CLASSE_VAZIA else rotulo
    cor = (0, 200, 0) if rotulo not in (CLASSE_VAZIA, "?") else (0, 200, 255)
    
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    
    # Imprimir a probabilidade media (se tiver "?" exibe prob maxima msm assim)
    prob_display = probs_media[i] if rotulo != "?" else np.max(probs_media)
    cv2.putText(frame, f"{texto} ({prob_display:.0%})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)
                
    for j, c in enumerate(class_names):
        y = 55 + j * 22
        cv2.rectangle(frame, (10, y), (10 + int(probs_media[j] * 150), y + 14), (0, 200, 255), -1)
        cv2.putText(frame, c, (170, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    cv2.putText(frame, f"t = {n / fps:5.1f}s", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
    if writer is not None:
        writer.write(frame)
        
    if webcam or os.environ.get("MOSTRAR", "1") == "1":
        cv2.imshow("Reconhecimento com CNN", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
            
    n += 1

cap.release()
if writer is not None:
    writer.release()
    print(f"Video anotado salvo em: {saida}")
cv2.destroyAllWindows()


# MELHORIA: Linha do tempo
print("\nLinha do tempo (Pokemon identificados):")
pokemon_encontrados = set()

if historico_timeline:
    inicio_idx = 0
    rotulo_atual = historico_timeline[0]
    
    for idx, r in enumerate(historico_timeline):
        if r != rotulo_atual:
            tempo_inicio = inicio_idx / fps
            tempo_fim = idx / fps
            duracao = tempo_fim - tempo_inicio
            
            # Filtro para > 0.5s e pular background ou indefinidos
            if rotulo_atual not in (CLASSE_VAZIA, "?") and duracao >= 0.5:
                print(f"  {tempo_inicio:6.1f}s a {tempo_fim:6.1f}s  ->  {rotulo_atual}")
                pokemon_encontrados.add(rotulo_atual)
                
            inicio_idx = idx
            rotulo_atual = r
            
    # Computar o ultimo segmento
    tempo_inicio = inicio_idx / fps
    tempo_fim = len(historico_timeline) / fps
    duracao = tempo_fim - tempo_inicio
    if rotulo_atual not in (CLASSE_VAZIA, "?") and duracao >= 0.5:
        print(f"  {tempo_inicio:6.1f}s a {tempo_fim:6.1f}s  ->  {rotulo_atual}")
        pokemon_encontrados.add(rotulo_atual)
        
print(f"Pokemon encontrados no video: {sorted(list(pokemon_encontrados))}")
