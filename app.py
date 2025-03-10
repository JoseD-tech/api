from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import io
import torch
import pathlib
from base64 import b64encode
import json

# Solución temporal para Windows (si es necesario)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Configuración de la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Colores para las clases detectadas
COLORS = {
    "RBC": (0, 255, 0),   # Verde
    "WBC": (255, 0, 0),   # Rojo
    "Platelet": (0, 0, 255)  # Azul
}

# Cargar el modelo YOLOv5 al iniciar la aplicación
def load_model():
    try:
        # Aplicar fix para Windows
        pathlib.PosixPath = pathlib.WindowsPath
        
        # Cargar modelo YOLOv5
        model = torch.hub.load(
            "ultralytics/yolov5", 
            "custom", 
            path="best.pt",
            force_reload=True
        )
        
        # Restaurar PosixPath
        pathlib.PosixPath = temp
        return model
    except Exception as e:
        raise RuntimeError(f"Error cargando el modelo: {str(e)}")

# Cargar el modelo al iniciar la aplicación
model = load_model()

# Función para procesar la imagen
def process_image(image, config=None):
    """Procesa la imagen y realiza las detecciones"""
    try:
        # Convertir a numpy array
        image_np = np.array(image)
        
        # Realizar detección
        results = model(image_np)
        detections = results.xyxy[0]
        
        # Crear imagen con bounding boxes
        image_with_boxes = image_np.copy()
        
        # Usar los parámetros de configuración o los valores por defecto
        line_thickness = config.get("line_thickness", 1) if config else 1
        alpha = config.get("alpha", 0.3) if config else 0.3
        font_scale = config.get("font_scale", 0.5) if config else 0.5
        font_thickness = config.get("font_thickness", 1) if config else 1
        
        # Dibujar detecciones
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            class_name = model.names[int(cls)]
            color = COLORS.get(class_name, (0, 255, 0))
            
            # Dibujar cuadro con transparencia
            overlay = image_with_boxes.copy()
            cv2.rectangle(
                overlay,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                int(line_thickness),  # Convertir a entero
            )
            cv2.addWeighted(overlay, alpha, image_with_boxes, 1 - alpha, 0, image_with_boxes)
            
            # Dibujar etiqueta
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                image_with_boxes,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                int(font_thickness),  # Convertir a entero
            )
        
        # Convertir la imagen de BGR a RGB
        image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        
        return image_with_boxes, detections
        
    except Exception as e:
        raise RuntimeError(f"Error procesando imagen: {str(e)}")

@app.route("/")
def index():
    return "Hola mundo"

# Endpoint para detectar objetos en una imagen
@app.route("/detect/", methods=["POST"])
def detect_objects():
    """Endpoint para detección de objetos en imágenes usando YOLOv5"""
    try:
        # Validar que se haya enviado un archivo
        if "file" not in request.files:
            return jsonify({"error": "No se proporcionó ningún archivo"}), 400
        
        file = request.files["file"]
        
        # Validar tipo de archivo
        if not file.content_type.startswith("image/"):
            return jsonify({"error": "Formato de archivo no soportado"}), 400
        
        # Leer y procesar imagen
        image = Image.open(file.stream)
        
        # Parsear la configuración si se proporciona
        config = None
        if "config" in request.form:
            config = json.loads(request.form["config"])
        
        # Procesar imagen
        processed_image, detections = process_image(image, config)
        
        # Convertir imagen a base64
        pil_image = Image.fromarray(processed_image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_base64 = b64encode(buffered.getvalue()).decode("utf-8")
        
        # Preparar respuesta
        detections_list = [
            {
                "class": model.names[int(det[5])],
                "confidence": float(det[4]),
                "bbox": {
                    "x1": float(det[0]),
                    "y1": float(det[1]),
                    "x2": float(det[2]),
                    "y2": float(det[3])
                }
            }
            for det in detections
        ]
        
        return jsonify({
            "detections": detections_list,
            "count": len(detections),
            "image_base64": img_base64,
            "format": "image/jpeg"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run()