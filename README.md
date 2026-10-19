# Predicción de Abandono de Clientes (Telco Churn) - Actividad 5

Este repositorio contiene el desarrollo técnico para la predicción de fuga de clientes en una empresa de telecomunicaciones, utilizando algoritmos de Machine Learning y gestionando el ciclo de vida de los experimentos con **MLflow**.

## 📂 Estructura del Repositorio

- `datos/datos_ini/`: Dataset original en bruto (`telco_churn.csv`).
- `datos/datos_limp/`: Dataset procesado, escalado y listo para modelado (`telco_churn_limpio.csv`).
- `fuentes/`: Scripts ejecutables del proyecto.
  - `datos_prep.py`: Pipeline de limpieza que maneja valores nulos (ej. en `TotalCharges`), aplica codificación *One-Hot* y estandariza variables numéricas.
  - `train.py`: Script de entrenamiento que compara Regresión Logística y Random Forest, busca hiperparámetros con `GridSearchCV` y registra los artefactos.

## ⚙️ Tecnologías y MLOps

- **Lenguaje:** Python
- **Machine Learning:** Scikit-Learn, Pandas, Seaborn
- **MLOps:** MLflow conectado a una base de datos local (`sqlite:///mlflow.db`) para el registro inmutable de parámetros, métricas de evaluación y almacenamiento de matrices de confusión.

## 📊 Resultados del Modelado

En el contexto de negocio del abandono de clientes, la métrica crítica a optimizar es el **Recall (Sensibilidad)**, ya que el error más costoso es dejar escapar a un cliente sin detectarlo (falso negativo).

- **Modelo Seleccionado:** Regresión Logística
- **Parámetros Óptimos:** `C = 10`, `solver = 'lbfgs'`
- **Desempeño:** Logró un Recall de **0.595** (superando al 0.429 del Random Forest) y mantuvo una Exactitud (Accuracy) global de 0.820.

## 📝 Changelog (Historial de Cambios)

- **v1.0:** Inicialización de la estructura de carpetas y carga del dataset crudo.
- **v1.1:** Creación del script `datos_prep.py` para la transformación y estandarización de los datos.
- **v1.2:** Desarrollo de `train.py` con implementación de modelos base y validación cruzada (K-Fold=5).
- **v1.3:** Integración de MLflow para el rastreo de experimentos y almacenamiento de artefactos; resolución de conectividad en el entorno de pruebas.
