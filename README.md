# UrbanSound8K Deep Learning Classifier

Este proyecto desarrolla, entrena y despliega un sistema de clasificación de sonidos urbanos utilizando el dataset **UrbanSound8K**. Se diseñan y comparan múltiples arquitecturas (Baseline, CNN, LSTM y CRNN) para seleccionar el modelo con mejor capacidad de generalización. Finalmente, el modelo final se empaqueta y despliega en **Amazon SageMaker** mediante un endpoint de inferencia en tiempo real.

---

## Objetivos del proyecto

- Preprocesar más de 8k audios en forma de **mel-espectrogramas normalizados**.  
- Implementar y entrenar diferentes arquitecturas de deep learning.  
- Elegir el mejor modelo según **validación** usando métricas sólidas.  
- Evaluar rigurosamente sobre el conjunto de prueba.  
- Exportar el modelo y sus artefactos para despliegue en SageMaker.  
- Crear un endpoint capaz de recibir audios reales y devolver predicciones confiables.

---


## Dataset

El dataset **UrbanSound8K** contiene 8732 clips de audio clasificados en 10 categorías:

- air_conditioner  
- car_horn  
- children_playing  
- dog_bark  
- drilling  
- engine_idling  
- gun_shot  
- jackhammer  
- siren  
- street_music  

Se sigue la partición oficial:  
- **fold1–8** → entrenamiento  
- **fold9** → validación  
- **fold10** → prueba  

---


# Preprocesamiento de Audio

El preprocesamiento es crítico porque transforma audios crudos en representaciones que los modelos pueden interpretar. Todo el pipeline se implementa en `PreprocesamientoDataSet.ipynb`.

### Pasos principales:

### **1. Carga y normalización temporal**
- Los audios se cargan en **mono** a **22050 Hz**.
- Todos se ajustan a una duración fija de **4 segundos** mediante padding o recorte.
- Esto garantiza que cada muestra produzca un tensor de tamaño consistente.

### **2. Extracción de mel-espectrogramas**
- Se utiliza `librosa` para calcular espectrogramas de **128 filtros mel**.
- Luego se convierten a dB para resaltar contrastes energéticos relevantes.
- El resultado es una matriz `(128, n_frames)` que resume contenido frecuencial + temporal.

### **3. Normalización global**
- La media y desviación estándar se calculan **solo con el conjunto de entrenamiento**.
- Cada espectrograma se normaliza usando esos valores.
- Este paso es obligatorio para evitar fugas de validación/test.

### **4. Artefactos guardados**
Se generan archivos que luego son usados:
- preprocessing_params.json
- label_encoder.pkl
- mean.npy
- std.npy


Con esto, el endpoint puede replicar exactamente el preprocesamiento en inferencia.

---

# Arquitecturas Utilizadas

En `EntrenamientoModelos.ipynb` se implementaron cuatro modelos con Keras:

### **1. Baseline (red densa)**
Modelo minimalista diseñado solo para establecer un punto de referencia. Usa *GlobalAveragePooling* sobre el espectrograma y capas densas.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def build_baseline(input_shape, num_classes):
    model = Sequential([
        GlobalAveragePooling2D(input_shape=input_shape),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(128, activation="relu"),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

baseline = build_baseline(input_shape, num_classes)
hist_base, ckpt_base = train_and_save(baseline, "baseline")
plot_history(hist_base, "Baseline")
```

### **2. CNN**
Arquitectura convolucional 2D que trabaja directamente sobre el mel-espectrograma como si fuera una imagen.  
Aprovecha:
- extracción jerárquica de patrones espectrales,  
- invariancia espacial,  
- filtros que capturan características como formantes, ataques, ruidos impulsivos, etc.

```
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu",padding='same', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation="relu",padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(256, (3,3), activation="relu", padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

cnn = build_cnn(input_shape, num_classes)
hist_cnn, ckpt_cnn = train_and_save(cnn, "cnn")
plot_history(hist_cnn, "CNN")

```

**→ Fue el modelo con mejor validación.**

### **3. LSTM**
Convierte el espectrograma en una secuencia de frames y aplica LSTM.  
Adecuado para dependencias temporales largas, pero más sensible al ruido y menos eficiente que CNN para capturar patrones espectro-espaciales.

```
from tensorflow.keras import layers, models

def build_lstm_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    mel_bins = input_shape[0]
    frames = input_shape[1]

    x = layers.Reshape((frames, mel_bins))(inp)
    x = layers.LSTM(128, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

lstm = build_lstm_model(input_shape, num_classes)
hist_lstm, ckpt_lstm = train_and_save(lstm, "lstm")
plot_history(hist_lstm, "LSTM")
```

### **4. CRNN**
Combinación de convoluciones + LSTM:  
- la parte CNN extrae patrones locales,  
- la parte LSTM captura dinámica temporal global.

```
def build_crnn(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(inp)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2,2))(x)

    shape = tf.keras.backend.int_shape(x)
    new_mels, new_frames, channels = shape[1], shape[2], shape[3]

    x = layers.Reshape((new_frames, new_mels * channels))(x)
    x = layers.LSTM(128)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

crnn = build_crnn(input_shape, num_classes)
hist_crnn, ckpt_crnn = train_and_save(crnn, "crnn")
plot_history(hist_crnn, "CRNN")
```

Aunque robusto, no superó a la CNN pura en este dataset.

---

# Resultados

### **Validación**
| Modelo | Val Accuracy |
|--------|--------------|
| Baseline | 0.2549 |
| LSTM | 0.5564 |
| CRNN | 0.6740 |
| **CNN** | **0.7377** |

**Métrica seleccionada: `val_accuracy`**  
Se eligió porque refleja la capacidad real del modelo para generalizar antes del test, evitando tomar decisiones basadas en un conjunto que no debe influir en la selección final.

### **Prueba final (Test)**
**Test accuracy: 0.6941**

El mejor desempeño se observó en clases transitorias y altamente distintivas. Clases con espectros más uniformes (como engine_idling) resultaron más desafiantes, lo cual es típico en UrbanSound8K.

---

# Despliegue en Amazon SageMaker

Implementado en `Endpoint.ipynb`.

### Etapas:

1. **Reconstrucción del modelo ganador (CNN)** y carga de pesos.
2. Exportación a formato **SavedModel**.
3. Empaquetado en `model.tar.gz` junto con:
   - mean.npy  
   - std.npy  
   - preprocessing_params.json  
   - deploy_metadata.json  
4. Carga del archivo al bucket S3.
5. Creación de un endpoint usando `TensorFlowModel`.
6. Cliente de predicción con preprocesamiento idéntico al usado en entrenamiento.

Se aplica una regla final de decisión:
Si probabilidad < 0.6 -> "indeterminado"

- Evita clasificaciones erróneas cuando el modelo está indeciso.  
- Reduce falsos positivos, especialmente en clases parecidas (engine_idling, air_conditioner).  
- Es un punto medio razonable entre sensibilidad y precisión, basado en experimentación previa.
  
<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/88c3e7f8-4718-4768-86a6-5239dda37412" />


---

# Inferencia Local y Remota

Implementado en `Predictor.ipynb`.

Este notebook permite probar el endpoint con audios externos reales y reproduce el mismo pipeline de preprocesamiento del entrenamiento:

### Preprocesamiento en inferencia:
- Carga con `torchaudio`.
- Resample a 22050 Hz.
- Conversión a mono.
- Extracción de mel-espectrograma.
- Conversión a dB.
- Normalización interna.
- Ajuste a `(128, 160, 1)` para compatibilidad con el modelo.

### Llamada al endpoint:
Se envía un JSON con la entrada en forma de tensor y se recuperan las predicciones del modelo desplegado.

### Ejemplos evaluados:
- **trafic.wav → drilling (0.648)**  
- **shotgun-kaliber-20-compressed-the-spike-by-12-db-wav-100826.wav → street_music (0.975)**  
- **Desert Eagle Single Shot Gunshot Sound Effect.mp3 → dog_bark (0.9999)**  

Esto evidencia el impacto del formato, del ruido ambiental y de la distribución real del dataset.

---

# Estructura del Proyecto

- **PreprocesamientoDataSet.ipynb** – Preprocesamiento completo y generación de artefactos.  
- **EntrenamientoModelos.ipynb** – Entrenamiento y comparación de las cuatro arquitecturas.  
- **Endpoint.ipynb** – Exportación + creación del endpoint.  
- **Predictor.ipynb** – Pruebas reales contra el endpoint.  
- **model.tar.gz** – Modelo final empaquetado.  
- **README.md** – Documentación del proyecto.

---

# Conclusiones

- La **CNN** demostró el mejor balance entre sesgo y varianza, convirtiéndose en el modelo óptimo para este dataset.  
- El preprocesamiento fue decisivo para estabilizar el entrenamiento y mejorar la separabilidad entre clases.  
- El desempeño final en test (≈ 0.69) es razonable considerando el ruido, diversidad y complejidad del dataset UrbanSound8K.  
- El endpoint implementado cumple los requisitos de producción, incluyendo la salida “indeterminado” para predicciones de baja confianza.  
- La modularidad del pipeline facilita futuras mejoras, reentrenamientos o reemplazo del modelo sin rehacer todo el sistema.
- No se aplicó data augmentation, lo cual podría ayudar en clases desbalanceadas.

---


