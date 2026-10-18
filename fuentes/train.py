import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os

def entrenar_y_registrar(nombre_modelo, modelo, param_grid, X_train, y_train, X_test, y_test):
    # Iniciar el registro en MLflow
    with mlflow.start_run(run_name=nombre_modelo):
        print(f"Entrenando {nombre_modelo} con GridSearchCV...")
        
        # Validación cruzada K-Fold = 5
        grid = GridSearchCV(modelo, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        mejor_modelo = grid.best_estimator_
        y_pred = mejor_modelo.predict(X_test)
        y_prob = mejor_modelo.predict_proba(X_test)[:, 1] if hasattr(mejor_modelo, "predict_proba") else y_pred
        
        # 1. Registrar Hiperparámetros
        mlflow.log_params(grid.best_params_)
        
        # 2. Registrar Métricas de Clasificación
        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }
        mlflow.log_metrics(metricas)
        
        # 3. Registrar Artefacto Gráfico (Matriz de Confusión)
        plt.figure(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusión - {nombre_modelo}')
        plt.ylabel('Real')
        plt.xlabel('Predicción')
        ruta_img = f"matriz_{nombre_modelo.replace(' ', '_')}.png"
        plt.savefig(ruta_img)
        plt.close()
        mlflow.log_artifact(ruta_img)
        
        # 4. Registrar Modelo
        mlflow.sklearn.log_model(
    mejor_modelo, 
    f"modelo_{nombre_modelo}", 
    serialization_format="cloudpickle"
)
        print(f"{nombre_modelo} finalizado. ROC-AUC: {metricas['roc_auc']:.4f}\n")

if __name__ == "__main__":
    # Cargar datos limpios
    df = pd.read_csv("datos/datos_limp/telco_churn_limpio.csv")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configurar MLflow para que guarde localmente en la carpeta del proyecto
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Telco_Churn_Experiment")

    # Modelo 1: Regresión Logística
    lr = LogisticRegression(max_iter=1000)
    lr_params = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}

    # Modelo 2: Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}

    entrenar_y_registrar("Regresion_Logistica", lr, lr_params, X_train, y_train, X_test, y_test)
    entrenar_y_registrar("Random_Forest", rf, rf_params, X_train, y_train, X_test, y_test)
