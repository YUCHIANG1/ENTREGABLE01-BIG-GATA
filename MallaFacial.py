import cv2
import mediapipe as mp
import math

# Inicializar MediaPipe
mpDibujo = mp.solutions.drawing_utils
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=2)  # Ajustar el número máximo de rostros a detectar

# Configuración de dibujo
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

# Iniciar la captura de video desde la cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Corrección de color de la imagen
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen con MediaPipe Face Mesh
    resultados = MallaFacial.process(frameRGB)

    # Si se detectan rostros en la imagen
    if resultados.multi_face_landmarks:
        for idx, rostro in enumerate(resultados.multi_face_landmarks):
            # Dibujar los puntos de referencia faciales y conexiones
            mpDibujo.draw_landmarks(frame, rostro, mpMallaFacial.FACEMESH_CONTOURS, ConfDibu, ConfDibu)

            # Obtener las dimensiones del frame
            h, w, c = frame.shape

            # Convertir las coordenadas normalizadas a píxeles
            x158 = int(rostro.landmark[158].x * w)
            y158 = int(rostro.landmark[158].y * h)
            x65 = int(rostro.landmark[65].x * w)
            y65 = int(rostro.landmark[65].y * h)
            x385 = int(rostro.landmark[385].x * w)
            y385 = int(rostro.landmark[385].y * h)
            x295 = int(rostro.landmark[295].x * w)
            y295 = int(rostro.landmark[295].y * h)
            x308 = int(rostro.landmark[308].x * w)
            y308 = int(rostro.landmark[308].y * h)
            x78 = int(rostro.landmark[78].x * w)
            y78 = int(rostro.landmark[78].y * h)
            x14 = int(rostro.landmark[14].x * w)
            y14 = int(rostro.landmark[14].y * h)
            x13 = int(rostro.landmark[13].x * w)
            y13 = int(rostro.landmark[13].y * h)

            # Calcular las longitudes de diferentes características faciales
            longitud1 = math.hypot(x158 - x65, y158 - y65)
            longitud2 = math.hypot(x385 - x295, y385 - y295)
            longitud3 = math.hypot(x308 - x78, y308 - y78)
            longitud4 = math.hypot(x14 - x13, y14 - y13)

            # Clasificación de emociones basada en las longitudes
            emocion = ""
            if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                emocion = 'Persona Enojada'
            elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 > 30:
                emocion = 'Persona Feliz'
            elif longitud1 > 35 and longitud2 > 35 and longitud3 > 80 and longitud3 < 90 and longitud4 > 20:
                emocion = 'Persona Asombrada'
            elif longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 80 and longitud3 < 100:
                emocion = 'Persona Triste'

            # Mostrar la etiqueta de la emoción en la imagen cerca de la cara
            # Usaremos las coordenadas del punto 10 (puedes cambiar este punto si lo deseas)
            x10 = int(rostro.landmark[10].x * w)
            y10 = int(rostro.landmark[10].y * h)
            cv2.putText(frame, emocion, (x10, y10 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar la imagen con el reconocimiento de emociones
    cv2.imshow("Reconocimiento de Emociones", frame)

    # Salir del bucle si se presiona la tecla 'Esc'
    if cv2.waitKey(1) == 27:
        break

# Liberar la captura de video y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
