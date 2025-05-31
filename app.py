import cv2
import os
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from PIL import Image
from datetime import datetime
import requests
import io

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

known_faces = {}
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Cargar rostros conocidos desde la carpeta base_rostros (recortados)
def load_known_faces():
    for filename in os.listdir("base_rostros"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join("base_rostros", filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(img_rgb)
            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                ih, iw, _ = img.shape
                x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                             int(bbox.width * iw), int(bbox.height * ih)
                rostro_recortado = img[y:y+h, x:x+w]
                rostro_recortado = cv2.resize(rostro_recortado, (150, 150))  # normalizar tamaño
                known_faces[filename] = rostro_recortado
            else:
                print(f"No se detectó rostro en {filename}")

load_known_faces()

# Comparar dos imágenes de rostro recortado
def is_match(face1, face2):
    face1 = cv2.resize(face1, (150, 150))
    face2 = cv2.resize(face2, (150, 150))
    diff = np.linalg.norm(face1.astype("float32") - face2.astype("float32"))
    return diff < 80  # umbral ajustable

# Detectar guiño usando MediaPipe FaceMesh
def detectar_guiño(imagen):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False

        landmarks = results.multi_face_landmarks[0].landmark

        right_eye_ids = [33, 159, 160, 158, 144, 153]
        left_eye_ids = [263, 386, 387, 385, 373, 380]

        def eye_aspect_ratio(eye_points):
            A = np.linalg.norm(np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y]) -
                               np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y]))
            B = np.linalg.norm(np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y]) -
                               np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y]))
            C = np.linalg.norm(np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y]) -
                               np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y]))
            ear = (A + B) / (2.0 * C)
            return ear

        left_ear = eye_aspect_ratio(left_eye_ids)
        right_ear = eye_aspect_ratio(right_eye_ids)

        if (left_ear < 0.22 and right_ear > 0.25) or (right_ear < 0.22 and left_ear > 0.25):
            return True

        return False

# Enviar mensaje por WhatsApp usando CallMeBot
def send_whatsapp_message(message):
    phone_number = "+51902697385"
    apikey = "2408114"
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone_number}&text={message}&apikey={apikey}"
    try:
        response = requests.get(url)
        print("Mensaje enviado:", response.text)
    except Exception as e:
        print("Error al enviar WhatsApp:", e)

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
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = face_detector.process(frame_rgb)

    if not results.detections:
        send_whatsapp_message("🚫 Timbre activado. No se detectó ningún rostro.")
        return jsonify({"resultado": "No se detectó ningún rostro"}), 200

    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), \
                     int(bbox.width * iw), int(bbox.height * ih)
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (150, 150))

        for name, known_face in known_faces.items():
            if is_match(face_img, known_face):
                if detectar_guiño(frame):
                    send_whatsapp_message(f"⚠️ Timbre activado. Rostro reconocido: {name}. Se detectó un GUIÑO (posible emergencia).")
                    return jsonify({"resultado": f"Rostro reconocido: {name}. GUIÑO detectado."}), 200
                else:
                    send_whatsapp_message(f"✅ Timbre activado. Rostro reconocido: {name}. Sin guiño.")
                    return jsonify({"resultado": f"Rostro reconocido: {name}. Sin guiño."}), 200

    send_whatsapp_message("❗ Timbre activado. Rostro NO reconocido.")
    return jsonify({"resultado": "Rostro no reconocido"}), 200

if __name__ == '__main__':
    os.makedirs("imagenes_recibidas", exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
