# Train YOLOv11

## Portfolio summary / Resumen de portafolio

Train_YOLOv11 automates YOLOv11 instance segmentation training and uses Optuna to search for better hyperparameters. It is designed for crack detection/segmentation experiments where repeatability and measurable training configuration matter.

Train_YOLOv11 automatiza el entrenamiento de segmentacion de instancias con YOLOv11 y utiliza Optuna para buscar mejores hiperparametros. Esta orientado a experimentos de deteccion/segmentacion de grietas donde importan la repetibilidad y la configuracion medible del entrenamiento.

**Technologies / Tecnologias:** Python, YOLOv11, Optuna, computer vision, segmentation.

**Project status / Estado:** Research/experimentation toolkit for model training.

**Next steps / Proximos pasos:** add an example run configuration, document expected dataset structure, and publish reproducible metrics.

---

# 🚀 Entrenamiento de YOLOv11 con Optimización Automática de Hiperparámetros

Este proyecto implementa un sistema de entrenamiento automatizado para modelos YOLOv11 (detección con bounding boxes o segmentación de instancias), utilizando **Optuna** para la optimización automática de hiperparámetros. El objetivo es entrenar un modelo de detección y/o segmentación de grietas en concreto.

## 📋 Descripción

Este proyecto está diseñado para:

- **Entrenar modelos YOLOv11** (detección con bounding boxes) o **YOLOv11-seg** (segmentación de instancias) para detectar grietas en imágenes de concreto
- **Optimizar automáticamente hiperparámetros** usando Optuna (Tree-structured Parzen Estimator)
- **Evaluar múltiples configuraciones** de forma sistemática y encontrar la mejor combinación de parámetros
- **Generar visualizaciones** de la evolución de las métricas durante la optimización

### Características principales:

- ✅ Optimización automática de hiperparámetros (tamaño de imagen, batch size, learning rate, optimizador, etc.)
- ✅ Manejo robusto de errores (especialmente errores de memoria GPU)
- ✅ Visualización de resultados con gráficos de evolución
- ✅ Resumen automático de las mejores configuraciones encontradas
- ✅ Soporte para GPU con limpieza automática de memoria

## 🏗️ Estructura del Proyecto

```
Train_YOLOv11/
├── train_yolo11.py          # Script principal de entrenamiento y optimización
├── data.yaml                 # Configuración del dataset (rutas y clases)
├── yolo11s-seg.pt           # Modelo base pre-entrenado (YOLOv11-seg small)
├── train/                    # Imágenes y etiquetas de entrenamiento
│   ├── images/
│   └── labels/
├── valid/                    # Imágenes y etiquetas de validación
│   ├── images/
│   └── labels/
├── test/                     # Imágenes y etiquetas de prueba
│   ├── images/
│   └── labels/
└── runs/                     # Resultados de entrenamiento
    └── optuna_search/        # Resultados de la optimización
```

## 🔧 Requisitos Previos

### 1. Python y Dependencias

Asegúrate de tener Python 3.8 o superior instalado. Luego instala las dependencias:

```bash
pip install ultralytics optuna pyyaml matplotlib torch torchvision
```

### 2. GPU (Recomendado)

Para entrenar modelos YOLOv11 de manera eficiente, se recomienda tener una GPU NVIDIA con CUDA instalado. El script detectará automáticamente si hay GPU disponible.

### 3. Dataset

El proyecto utiliza un dataset de detección de grietas en concreto con:
- **3,717 imágenes** de entrenamiento
- **200 imágenes** de validación
- **112 imágenes** de prueba
- **1 clase**: `crack` (grieta)

## 📝 Configuración

### Archivo `data.yaml`

Este archivo define la estructura del dataset:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['crack']
```

### Parámetros en `train_yolo11.py`

Puedes modificar estos parámetros según tus necesidades:

```python
DATASET = "data.yaml"               # Ruta al YAML del dataset
BASE_MODEL = "yolo11s.pt"           # Modelo base: "yolo11s.pt" (bbox) o "yolo11s-seg.pt" (seg)
PROJECT = "runs/optuna_search"      # Carpeta de resultados
DEVICE = "0"                        # GPU ("0", "1", etc.) o "cpu"
EPOCHS = 100                        # Épocas por prueba
WORKERS = 8                         # Hilos del dataloader
N_TRIALS = 20                       # Número de combinaciones a probar
```

**Nota sobre tipos de modelo:**
- **Bounding Boxes (Detección)**: Usa modelos sin `-seg` (ej: `yolo11s.pt`) - Detecta objetos con rectángulos
- **Segmentación**: Usa modelos con `-seg` (ej: `yolo11s-seg.pt`) - Detecta objetos con máscaras de píxeles

### Hiperparámetros que se optimizan automáticamente:

- **Tamaño de imagen** (`imgsz`): 416, 512, 640 píxeles
- **Batch size** (`batch`): 8, 16, 32
- **Learning rate inicial** (`lr0`): 1e-4 a 1e-2 (escala logarítmica)
- **Optimizador**: SGD, Adam, AdamW
- **Momentum**: 0.7 a 0.99
- **Weight decay**: 1e-6 a 1e-3 (escala logarítmica)

## 🚀 Pasos para Ejecutar

### Paso 1: Preparar el Entorno

1. Clona o descarga este repositorio
2. Asegúrate de tener todas las dependencias instaladas (ver sección de Requisitos)

### Paso 2: Verificar el Dataset

Asegúrate de que las carpetas `train/`, `valid/` y `test/` contengan:
- Imágenes en formato JPG/PNG
- Archivos de etiquetas en formato YOLO (.txt) con el mismo nombre que las imágenes

### Paso 3: Verificar el Modelo Base

El archivo `yolo11s-seg.pt` debe estar en el directorio raíz. Si no lo tienes, Ultralytics lo descargará automáticamente la primera vez que ejecutes el script.

### Paso 4: Ejecutar el Entrenamiento

```bash
python train_yolo11.py
```

### Paso 5: Monitorear el Progreso

Durante la ejecución verás:

- ✅ Progreso de cada trial (prueba de hiperparámetros)
- 📊 Métricas en tiempo real (mAP50-95, Recall, Precision)
- ⚠️ Advertencias si algún trial falla (por ejemplo, por falta de memoria GPU)
- 🏁 Resumen final con las mejores configuraciones

### Paso 6: Revisar los Resultados

Los resultados se guardan en `runs/optuna_search/`:

- **`optuna_best.yaml`**: Mejores hiperparámetros encontrados
- **`optuna_results_plot.png`**: Gráfico de evolución de métricas
- **`trial_X_*/`**: Carpetas individuales con resultados de cada trial
- **`trial_X_*/weights/best.pt`**: Mejor modelo de cada trial

## 🔍 Cómo Funciona

### 1. Proceso de Optimización

El script utiliza **Optuna** con el algoritmo **TPE (Tree-structured Parzen Estimator)**:

1. **Inicialización**: Crea un estudio de optimización que busca maximizar el mAP50-95
2. **Búsqueda**: Para cada trial, Optuna sugiere una combinación de hiperparámetros
3. **Entrenamiento**: Se entrena el modelo YOLOv11 con esos hiperparámetros
4. **Evaluación**: Se calcula el mAP50-95 del modelo entrenado
5. **Aprendizaje**: Optuna aprende de los resultados y sugiere mejores combinaciones
6. **Repetición**: Se repite el proceso hasta completar `N_TRIALS` trials

### 2. Manejo de Errores

El script incluye manejo robusto de errores:

- **Errores de memoria GPU**: Si un trial falla por falta de memoria, se registra y continúa con el siguiente
- **Limpieza automática**: Se limpia la memoria GPU después de cada trial
- **Pruning**: Trials que no mejoran se detienen tempranamente (MedianPruner)

### 3. Métricas Evaluadas

- **mAP50-95**: Mean Average Precision (promedio de mAP@0.5 a mAP@0.95) - **métrica principal**
- **Recall**: Tasa de detección de objetos
- **Precision**: Precisión de las detecciones

### 4. Visualización

Al finalizar, se genera un gráfico que muestra:
- Evolución del mAP50-95 por trial
- Evolución del Recall por trial
- Identificación visual de las mejores configuraciones


## ⚙️ Personalización

### Cambiar el Modelo Base

Puedes usar diferentes modelos pre-entrenados según tu necesidad:

**Modelos de Bounding Boxes (Detección):**
- `yolo11n.pt` (nano - más rápido, menos preciso)
- `yolo11s.pt` (small - balanceado) ⭐ **Recomendado para detección**
- `yolo11m.pt` (medium - más preciso, más lento)
- `yolo11l.pt` (large - muy preciso, muy lento)
- `yolo11x.pt` (xlarge - máximo rendimiento)

**Modelos de Segmentación:**
- `yolo11n-seg.pt` (nano - más rápido, menos preciso)
- `yolo11s-seg.pt` (small - balanceado) ⭐ **Recomendado para segmentación**
- `yolo11m-seg.pt` (medium - más preciso, más lento)
- `yolo11l-seg.pt` (large - muy preciso, muy lento)
- `yolo11x-seg.pt` (xlarge - máximo rendimiento)

### Ajustar el Espacio de Búsqueda

En la función `objective()`, puedes modificar los rangos de hiperparámetros:

```python
imgsz = trial.suggest_categorical("imgsz", [416, 512, 640, 768])  # Agregar más opciones
batch = trial.suggest_categorical("batch", [4, 8, 16, 32])        # Ajustar según tu GPU
lr0 = trial.suggest_float("lr0", 1e-5, 1e-1, log=True)           # Ampliar rango
```

### Usar CPU en lugar de GPU

Cambia `DEVICE = "cpu"` en la configuración (será mucho más lento).

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"

- Reduce el `batch` size en el espacio de búsqueda
- Reduce el `imgsz` (tamaño de imagen)
- Cierra otras aplicaciones que usen GPU
- Reduce `WORKERS`

### Error: "No se encontró results.yaml"

- Verifica que el entrenamiento se completó correctamente
- Revisa los logs del trial específico
- Aumenta `EPOCHS` si el entrenamiento se interrumpe muy temprano

### El proceso es muy lento

- Reduce `N_TRIALS` para menos pruebas
- Reduce `EPOCHS` para entrenamientos más cortos (aunque menos precisos)
- Usa un modelo más pequeño (`yolo11n-seg.pt`)
- Asegúrate de estar usando GPU

## 📚 Recursos Adicionales

- [Documentación de Ultralytics YOLOv11](https://docs.ultralytics.com/)
- [Documentación de Optuna](https://optuna.readthedocs.io/)
- [Dataset original en Roboflow](https://universe.roboflow.com/university-bswxt/crack-bphdr)

## 📄 Licencia

Este proyecto utiliza un dataset de dominio público. Consulta los archivos `README.roboflow.txt` y `README.dataset.txt` para más información sobre el dataset.

## 🤝 Contribuciones

Las mejoras y sugerencias son bienvenidas. Algunas ideas:

- Agregar más hiperparámetros a optimizar
- Implementar early stopping más sofisticado
- Agregar visualización de importancia de hiperparámetros
- Exportar modelos en diferentes formatos (ONNX, TensorRT, etc.)

---

**¡Feliz entrenamiento! 🎯**

