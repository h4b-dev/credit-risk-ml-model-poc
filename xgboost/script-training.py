# Importar librerías necesarias
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. Cargar el CSV con la data necesaria
# Sustituye 'data.csv' por la ruta real de tu archivo CSV
columns_to_use = ['total_transactions', 'total_amount_spent', 'on_time_payments', 'late_payments', 'bnpl_used', 'bnpl_default', 'target']
df = pd.read_csv('data/buyers_data_experimental.csv', usecols=columns_to_use)

# 2. Preparación de datos
# Supongamos que tu columna de target (score o aptitud para BNPL) se llama 'target'
# X son las características (features), y es el objetivo (target)
X = df.drop('target', axis=1)  # Elimina la columna de la variable target
y = df['target']  # La columna target es la variable dependiente

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el modelo XGBoost
# Configuración inicial de los hiperparámetros
model = xgb.XGBClassifier(
    n_estimators=100,      # Número de árboles
    max_depth=6,           # Profundidad máxima de los árboles
    learning_rate=0.1,     # Tasa de aprendizaje
    subsample=0.8,         # Proporción de datos usados para entrenar cada árbol
    colsample_bytree=0.8,  # Proporción de características usadas en cada árbol
    random_state=42
)

# Entrenar el modelo con el conjunto de datos de entrenamiento
model.fit(X_train, y_train)

# 4. Realizar predicciones
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidades de clase positiva

# 5. Evaluar el rendimiento del modelo
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Clasificación detallada (precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC para medir el área bajo la curva ROC
roc_auc = roc_auc_score(y_test, y_proba)
print(f'ROC-AUC: {roc_auc}')

# 6. Optimización de hiperparámetros usando GridSearchCV (opcional)
# Este paso es opcional si deseas mejorar el rendimiento del modelo
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='roc_auc', cv=3)
grid_search.fit(X_train, y_train)

# Mostrar los mejores hiperparámetros encontrados
print("Mejores parámetros encontrados: ", grid_search.best_params_)

# Entrenar el modelo con los mejores parámetros (opcional)
best_model = grid_search.best_estimator_

# 7. Calcular el score del cliente
# Asignamos un score basado en la probabilidad de la clase positiva (aptitud)
# Normalizamos entre un rango de 300 a 850, similar a sistemas de crédito tradicionales
min_score = 300
max_score = 850

# Normalización de las probabilidades a un rango de score
y_test_proba = best_model.predict_proba(X_test)[:, 1]
y_test_score = min_score + (y_test_proba * (max_score - min_score))

# Mostrar los scores calculados para los clientes del conjunto de prueba
df_test = X_test.copy()
df_test['calculated_score'] = y_test_score
print(df_test[['calculated_score']].head())

# Si deseas guardar el modelo para su uso posterior:
import joblib
joblib.dump(best_model, 'output/xgboost_credit_score_model.pkl')