import cv2
import os
import numpy as np
from flask import Flask, request, jsonify
import mediapipe as mp
from PIL import Image
from datetime import datetime

app = Flask(__name__)

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

known_faces = {}
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Cargar rostros conocidos desde la carpeta base_rostros
def load_known_faces():
    for filename in os.listdir("base_rostros"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join("base_rostros", filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detector.process(img_rgb)
            if results.detections:
                known_faces[filename] = img_rgb
            else:
                print(f"No se detectó rostro en {filename}")

load_known_faces()

# Comparar dos imágenes usando histogramas como método simple
def is_match(face1, face2):
    hist1 = cv2.calcHist([face1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2], [0], None, [256], [0, 256])
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score > 0.8

@app.route('/reconocer', methods=['POST'])
def reconocer():
    if 'foto' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files['foto']
    img = Image.open(file.stream).convert('RGB')
    frame = np.array(img)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = face_detector.process(frame_rgb)

    if not results.detections:
        return jsonify({"resultado": "No se detectó ningún rostro"}), 200

    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih)
        face_img = frame[y:y+h, x:x+w]

        for name, known_img in known_faces.items():
            if is_match(face_img, known_img):
                return jsonify({"resultado": f"Rostro reconocido: {name}"}), 200

    return jsonify({"resultado": "Rostro no reconocido"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
