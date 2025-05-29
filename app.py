from flask import Flask, request
import os
import cv2
import numpy as np
import mediapipe as mp
import base64
from datetime import datetime
import requests

app = Flask(__name__)
mp_face_detection = mp.solutions.face_detection

BASE_DIR = "base_rostros"
WHATSAPP_API_URL = "https://api.callmebot.com/whatsapp.php"
TO_NUMBER = "+51902697385"
API_KEY = "2408114"

def enviar_whatsapp(mensaje):
    params = {
        "phone": TO_NUMBER,
        "text": mensaje,
        "apikey": API_KEY
    }
    requests.get(WHATSAPP_API_URL, params=params)

def detectar_rostro_y_guiño(imagen):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        resultados = detector.process(imagen)
        if resultados.detections:
            return True  # Rostro detectado
    return False

def cargar_base_rostros():
    rostros = []
    for archivo in os.listdir(BASE_DIR):
        ruta = os.path.join(BASE_DIR, archivo)
        img = cv2.imread(ruta)
        if img is not None:
            rostros.append((archivo, img))
    return rostros

@app.route("/recibir", methods=["POST"])
def recibir_foto():
    if "imagen" not in request.files:
        return "No se envió imagen", 400

    file = request.files["imagen"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return "Imagen inválida", 400

    rostro_detectado = detectar_rostro_y_guiño(frame)
    if rostro_detectado:
        mensaje = f"✅ Rostro detectado - {datetime.now().strftime('%H:%M:%S')}"
    else:
        mensaje = f"⚠️ Alerta: No se detectó rostro - {datetime.now().strftime('%H:%M:%S')}"

    enviar_whatsapp(mensaje)
    return mensaje, 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)