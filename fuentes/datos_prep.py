import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def limpiar_y_preparar_datos(ruta_entrada, ruta_salida):
    print("Cargando datos originales...")
    df = pd.read_csv(ruta_entrada)
    
    # 1. Manejo de nulos ocultos en 'TotalCharges'
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 2. Eliminar identificadores que no sirven para predecir
    df = df.drop('customerID', axis=1)
    
    # 3. Convertir la variable objetivo (Churn) a 0 y 1
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # 4. Codificación de variables categóricas (One-Hot Encoding)
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    df_procesado = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)
    
    # 5. Escalado de variables numéricas continuas
    columnas_numericas = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df_procesado[columnas_numericas] = scaler.fit_transform(df_procesado[columnas_numericas])
    
    # Guardar el dataset limpio
    df_procesado.to_csv(ruta_salida, index=False)
    print(f"Datos limpios y transformados guardados exitosamente en: {ruta_salida}")

if __name__ == "__main__":
    # Estas rutas asumen que el script se corre desde la raíz del proyecto
    RUTA_ORIGINAL = "datos/datos_ini/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    RUTA_LIMPIA = "datos/datos_limp/telco_churn_limpio.csv"
    
    limpiar_y_preparar_datos(RUTA_ORIGINAL, RUTA_LIMPIA)
