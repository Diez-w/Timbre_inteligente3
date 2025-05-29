
from flask import Flask, request
import cv2
import numpy as np
import os
from deepface import DeepFace
import mediapipe as mp
import requests

app = Flask(__name__)

# Configuración de WhatsApp (CallMeBot)
WHATSAPP_NUMBER = '+51902697385'
API_KEY = '2408114'

# Ruta base donde están las imágenes de referencia
BASE_PATH = 'base_rostros'

# Inicializar MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Cargar rostros registrados
def cargar_rostros_base():
    rostros_db = {}
    for archivo in os.listdir(BASE_PATH):
        if archivo.endswith(('.jpg', '.png')):
            nombre = os.path.splitext(archivo)[0]
            path = os.path.join(BASE_PATH, archivo)
            rostros_db[nombre] = cv2.imread(path)
    return rostros_db

rostros_db = cargar_rostros_base()

# Detección de guiño
def detectar_guiño(imagen):
    rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    resultados = face_mesh.process(rgb)
    if not resultados.multi_face_landmarks:
        return False

    for rostro in resultados.multi_face_landmarks:
        ojo_izq = [rostro.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
        ojo_der = [rostro.landmark[i] for i in [33, 160, 158, 133, 153, 144]]

        def apertura_ojo(ojo):
            vertical = np.linalg.norm(np.array([ojo[1].x, ojo[1].y]) - np.array([ojo[5].x, ojo[5].y]))
            horizontal = np.linalg.norm(np.array([ojo[0].x, ojo[0].y]) - np.array([ojo[3].x, ojo[3].y]))
            return vertical / horizontal if horizontal != 0 else 0

        apertura_izq = apertura_ojo(ojo_izq)
        apertura_der = apertura_ojo(ojo_der)

        if abs(apertura_izq - apertura_der) > 0.15:
            return True
    return False

# Enviar mensaje por WhatsApp
def enviar_mensaje(mensaje):
    url = f'https://api.callmebot.com/whatsapp.php?phone={WHATSAPP_NUMBER}&text={mensaje}&apikey={API_KEY}'
    try:
        requests.get(url)
    except Exception as e:
        print(f"❌ Error al enviar WhatsApp: {e}")

@app.route('/')
def index():
    return 'Servidor activo ✅'

@app.route('/recibir', methods=['POST'])
def recibir():
    if 'imagen' not in request.files:
        return {'error': 'No se envió la imagen'}, 400

    archivo = request.files['imagen']
    img_np = np.frombuffer(archivo.read(), np.uint8)
    imagen = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    try:
        resultado = DeepFace.find(img_path=imagen, db_path=BASE_PATH, enforce_detection=False)
        if resultado[0].shape[0] > 0:
            nombre = resultado[0].iloc[0]['identity'].split(os.sep)[-1].split('.')[0]
            guiño = detectar_guiño(imagen)
            if guiño:
                mensaje = f"⚠️ ALARMA: Posible emergencia detectada en {nombre.upper()} (guiño detectado)"
                enviar_mensaje(mensaje)
                return {'estado': 'alerta', 'persona': nombre}
            else:
                mensaje = f"✅ Acceso autorizado para {nombre.upper()} (sin guiño)"
                enviar_mensaje(mensaje)
                return {'estado': 'autorizado', 'persona': nombre}
        else:
            return {'estado': 'desconocido'}, 200
    except Exception as e:
        print("Error en reconocimiento:", e)
        return {'error': 'Fallo en el procesamiento'}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
