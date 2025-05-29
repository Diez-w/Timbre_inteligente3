import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
import requests

app = Flask(__name__)

# Configuración de WhatsApp con CallMeBot
WHATSAPP_API_URL = "https://api.callmebot.com/whatsapp.php"
PHONE_NUMBER = "+51902697385"
API_KEY = "2408114"

# Inicializa MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)

# Carga de imágenes de rostros registrados
known_encodings = []
known_names = []

def encode_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
        return np.array(coords).flatten()
    return None

def load_known_faces():
    for filename in os.listdir("base_rostros"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join("base_rostros", filename)
            img = cv2.imread(img_path)
            encoding = encode_face(img)
            if encoding is not None:
                known_encodings.append(encoding)
                known_names.append(filename.split(".")[0])

def compare_faces(encoding):
    for i, known_encoding in enumerate(known_encodings):
        distance = np.linalg.norm(known_encoding - encoding)
        if distance < 0.08:  # Umbral ajustado para coincidencia
            return known_names[i]
    return None

def detectar_guiño(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return False

    landmarks = results.multi_face_landmarks[0].landmark

    # Índices para el ojo derecho y ojo izquierdo
    eye_indices = {
        "left": [33, 159],    # superior, inferior ojo izquierdo
        "right": [362, 386]   # superior, inferior ojo derecho
    }

    def ojo_cerrado(p1, p2):
        y1 = landmarks[p1].y
        y2 = landmarks[p2].y
        return abs(y1 - y2) < 0.015

    izquierdo = ojo_cerrado(*eye_indices["left"])
    derecho = ojo_cerrado(*eye_indices["right"])

    # Detectar si uno está cerrado y el otro abierto
    return (izquierdo and not derecho) or (derecho and not izquierdo)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    encoding = encode_face(image)
    if encoding is None:
        return jsonify({'status': 'No se detectó rostro'}), 400

    nombre = compare_faces(encoding)
    guiño = detectar_guiño(image)

    if nombre and guiño:
        mensaje = f"⚠️ Emergencia detectada en {nombre}. Se detectó un guiño sospechoso."
    elif nombre:
        mensaje = f"✅ Acceso autorizado para {nombre}. Sin signos de emergencia."
    else:
        mensaje = "❌ Rostro no reconocido. Acceso denegado."

    # Enviar mensaje por WhatsApp
    requests.get(WHATSAPP_API_URL, params={
        "phone": PHONE_NUMBER,
        "text": mensaje,
        "apikey": API_KEY
    })

    return jsonify({'status': mensaje})

if __name__ == '__main__':
    load_known_faces()
    app.run(host='0.0.0.0', port=10000)
