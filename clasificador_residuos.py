from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar MobileNetV2 preentrenado
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congelar las capas base del modelo preentrenado
base_model.trainable = False

# Añadir capas de clasificación
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax') 
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Preparar el generador de datos
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normaliza las imágenes y separa el 20% para validación

# Cargar el dataset desde el directorio
train_generator = datagen.flow_from_directory(
    'C:\\Users\\betom\\Desktop\\ProyectoDeteccion\\Garbage classification\\Garbage classification',  # Reemplaza con la ruta donde descargaste el dataset
    target_size=(224, 224),  # Asegúrate de que coincida con el tamaño de entrada esperado por MobileNetV2
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Usa el 80% de las imágenes para entrenamiento
)

validation_generator = datagen.flow_from_directory(
    'C:\\Users\\betom\\Desktop\\ProyectoDeteccion\\Garbage classification\\Garbage classification',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Usa el 20% de las imágenes para validación
)

# Entrenar el modelo
history = model.fit(
    train_generator,
    epochs=10,  
    validation_data=validation_generator
)

# Guardar el modelo entrenado
model.save('modelo_residuos_entrenado_5_clases.h5')

