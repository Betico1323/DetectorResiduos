import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Button, Label

# Cargar el modelo entrenado de clasificación de residuos
model = load_model('C:\\Users\\betom\\Desktop\\ProyectoDeteccion\\modelo_residuos_entrenado_5_clases.h5') 

# Definir las clases de residuos
classes = ['Metal', 'Vidrio', 'Plástico', 'Cartón', 'Papel']  

# Variable global para mostrar la predicción en la interfaz
predicted_label = ""

# Función para preprocesar la imagen
def preprocess_image(image):
    img_resized = cv2.resize(image, (224, 224))  # Redimensionar según el tamaño esperado por el modelo
    img_array = np.array(img_resized) / 255.0    # Normalizar
    img_array = np.expand_dims(img_array, axis=0) 
    return img_array

# Función para hacer predicciones
def predict_waste(image, model):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return classes[predicted_class]

# Función para actualizar la predicción en la interfaz gráfica
def update_prediction_text(label):
    global predicted_label
    predicted_label = label
    prediction_label.config(text=f'Residuo detectado: {predicted_label}')

# Función para capturar y predecir el residuo
def capture_and_predict():
    ret, frame = cap.read()
    if ret:
        predicted_label = predict_waste(frame, model)
        update_prediction_text(predicted_label)

# Función para cerrar la cámara y la ventana
def close_app():
    cap.release()
    cv2.destroyAllWindows()
    root.quit()

# Configurar la cámara
cap = cv2.VideoCapture(0)

# Verificar si la cámara está abierta
if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Configurar la ventana principal
root = tk.Tk()
root.title("Detección de Residuos")
root.geometry("800x600")

# Mostrar el video en un label dentro de la interfaz
video_label = Label(root)
video_label.pack()

# Mostrar la predicción en la interfaz
prediction_label = Label(root, text="Residuos detectados: Ninguno", font=("Arial", 14))
prediction_label.pack(pady=20)

# Botón para capturar imagen y predecir
capture_button = Button(root, text="Capturar y Predecir", command=capture_and_predict, font=("Arial", 12))
capture_button.pack(pady=10)

# Botón para cerrar la aplicación
quit_button = Button(root, text="Salir", command=close_app, font=("Arial", 12), bg="red", fg="white")
quit_button.pack(pady=10)

# Función para actualizar el video en tiempo real
def update_video():
    ret, frame = cap.read()
    if ret:
        # Convertir de BGR a RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (640, 480))
        img = cv2.flip(img, 1)  

        # Convertir a formato compatible con Tkinter
        from PIL import Image, ImageTk
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Llamar de nuevo a la función después de 10ms para un bucle continuo
    video_label.after(10, update_video)


# Iniciar la actualización del video en tiempo real
update_video()

# Iniciar la interfaz
root.mainloop()

# Liberar recursos
cap.release()
cv2.destroyAllWindows()