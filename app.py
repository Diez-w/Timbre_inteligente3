import cv2
import os
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from PIL import Image
import requests
import io

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

known_faces = {}  # key: nombre_archivo, value: landmarks (np.array)

face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Obtener landmarks faciales
def get_face_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

# Comparar dos rostros por landmarks
def is_match(input_landmarks, known_landmarks):
    if input_landmarks is None or known_landmarks is None:
        return False
    if len(input_landmarks) != len(known_landmarks):
        return False
    dist = np.linalg.norm(input_landmarks - known_landmarks)
    return dist < 0.12  # Umbral ajustable (cuanto menor, más estricta la comparación)

# Cargar imágenes base
def load_known_faces():
    for filename in os.listdir("base_rostros"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join("base_rostros", filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
            landmarks = get_face_landmarks(img)
            if landmarks is not None:
                known_faces[filename] = landmarks
            else:
                print(f"No se detectaron puntos faciales en {filename}")

load_known_faces()

# Detectar guiño
def detectar_guiño(imagen):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark
        right_eye_ids = [33, 159, 160, 158, 144, 153]
        left_eye_ids = [263, 386, 387, 385, 373, 380]

        def ear(eye_points):
            A = np.linalg.norm(np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y]) -
                               np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y]))
            B = np.linalg.norm(np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y]) -
                               np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y]))
            C = np.linalg.norm(np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y]) -
                               np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y]))
            return (A + B) / (2.0 * C)

        left_ear = ear(left_eye_ids)
        right_ear = ear(right_eye_ids)

        return (left_ear < 0.22 and right_ear > 0.25) or (right_ear < 0.22 and left_ear > 0.25)

# WhatsApp via CallMeBot
def send_whatsapp_message(message):
    phone_number = "+51902697385"
    apikey = "2408114"
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone_number}&text={message}&apikey={apikey}"
    try:
        response = requests.get(url)
        print("Mensaje enviado:", response.text)
    except Exception as e:
        print("Error al enviar WhatsApp:", e)

# Ruta principal
@app.route('/recibir', methods=['POST'])
def recibir():
    raw_image = request.get_data()
    if not raw_image:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    try:
        img = Image.open(io.BytesIO(raw_image)).convert('RGB')
    except Exception as e:
        return jsonify({"error": "Error al procesar imagen"}), 400

    frame = np.array(img)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = face_detector.process(frame_bgr)
    if not results.detections:
        send_whatsapp_message("🚫 Timbre activado. No se detectó ningún rostro.")
        return jsonify({"resultado": "No se detectó ningún rostro"}), 200

    input_landmarks = get_face_landmarks(frame_bgr)
    for name, known_landmarks in known_faces.items():
        if is_match(input_landmarks, known_landmarks):
            if detectar_guiño(frame_bgr):
                send_whatsapp_message(f"⚠️ Timbre activado. Rostro reconocido: {name}. Se detectó un GUIÑO (emergencia).")
                return jsonify({"resultado": f"Rostro reconocido: {name}. GUIÑO detectado."}), 200
            else:
                send_whatsapp_message(f"✅ Timbre activado. Rostro reconocido: {name}. Sin guiño.")
                return jsonify({"resultado": f"Rostro reconocido: {name}. Sin guiño."}), 200

    send_whatsapp_message("❗ Timbre activado. Rostro NO reconocido.")
    return jsonify({"resultado": "Rostro no reconocido"}), 200

if __name__ == '__main__':
    os.makedirs("imagenes_recibidas", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
