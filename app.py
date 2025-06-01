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

face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

known_faces = {}

def preprocess_image(img):
    """Mejora imágenes de baja calidad del ESP32-CAM"""
    # Convertir a escala de grises para el procesamiento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Ecualización de histograma para mejorar contraste
    gray = cv2.equalizeHist(gray)
    
    # Reducción de ruido (ajustar según necesidad)
    gray = cv2.medianBlur(gray, 3)
    
    # Convertir de vuelta a BGR para MediaPipe
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def extract_landmarks(img):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

def load_known_faces():
    print("Cargando rostros conocidos...")
    for filename in os.listdir("base_rostros"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join("base_rostros", filename)
            img = cv2.imread(img_path)
            img = preprocess_image(img)  # Preprocesar imágenes de referencia también
            landmarks = extract_landmarks(img)
            if landmarks is not None:
                known_faces[filename] = landmarks
                print(f"Cargado: {filename} con {landmarks.shape[0]} landmarks")
            else:
                print(f"No se detectó rostro en {filename}")

def is_match(input_landmarks, known_landmarks, threshold=0.15):
    if input_landmarks is None or known_landmarks is None:
        return False
    
    # Normalización de landmarks
    def normalize_landmarks(landmarks):
        centroid = np.mean(landmarks, axis=0)
        normalized = landmarks - centroid
        # Escala basada en la distancia entre los ojos (índices 133 y 362 en MediaPipe)
        eye_dist = np.linalg.norm(normalized[133] - normalized[362])
        if eye_dist > 0:
            normalized /= eye_dist
        return normalized
    
    input_norm = normalize_landmarks(input_landmarks)
    known_norm = normalize_landmarks(known_landmarks)
    
    # Solo comparamos landmarks clave (ojos, nariz, boca)
    key_indices = [33, 133, 362, 263, 1, 4, 5, 195, 197]  # Puntos faciales importantes
    dist = np.mean(np.linalg.norm(input_norm[key_indices] - known_norm[key_indices], axis=1))
    
    print(f"Distancia normalizada: {dist:.4f} (Umbral: {threshold})")
    return dist < threshold

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

        # Ajusta estos umbrales si es necesario
        if (left_ear < 0.22 and right_ear > 0.25) or (right_ear < 0.22 and left_ear > 0.25):
            return True

        return False

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
    frame = preprocess_image(frame)  # Aplicar preprocesamiento
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame_rgb)

    if not results.detections:
        send_whatsapp_message("🚫 Timbre activado. No se detectó ningún rostro.")
        print("No se detectó rostro en la imagen recibida.")
        return jsonify({"resultado": "No se detectó ningún rostro"}), 200

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        face_img = frame[y:y+h, x:x+w]

        print(f"Rostro detectado: bbox=({x},{y},{w},{h}), tamaño recorte={face_img.shape}")

        input_landmarks = extract_landmarks(frame)  # Procesar imagen completa
        if input_landmarks is None:
            print("No se pudo extraer landmarks del rostro detectado")
            continue

        for name, known_landmarks in known_faces.items():
            print(f"Comparando con rostro conocido: {name}")
            if is_match(input_landmarks, known_landmarks):
                if detectar_guiño(frame_rgb):
                    send_whatsapp_message(f"⚠️ Timbre activado. Rostro reconocido: {name}. Se detectó un GUIÑO (posible emergencia).")
                    print(f"GUIÑO detectado para {name}")
                    return jsonify({"resultado": f"Rostro reconocido: {name}. GUIÑO detectado."}), 200
                else:
                    send_whatsapp_message(f"✅ Timbre activado. Rostro reconocido: {name}. Sin guiño.")
                    print(f"Rostro reconocido sin guiño: {name}")
                    return jsonify({"resultado": f"Rostro reconocido: {name}. Sin guiño."}), 200

    send_whatsapp_message("❗ Timbre activado. Rostro NO reconocido.")
    print("No se reconoció ningún rostro después de comparar con la base.")
    return jsonify({"resultado": "Rostro no reconocido"}), 200

if __name__ == '__main__':
    os.makedirs("imagenes_recibidas", exist_ok=True)
    load_known_faces()
    app.run(host='0.0.0.0', port=5000)
