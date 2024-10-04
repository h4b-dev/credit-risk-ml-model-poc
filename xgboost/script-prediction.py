import numpy as np
import joblib

# 1. Cargar el modelo previamente entrenado
# Asegúrate de que el archivo 'xgboost_credit_score_model.pkl' esté en el mismo directorio o proporciona la ruta correcta
model_path = 'output/xgboost_credit_score_model.pkl'
model = joblib.load(model_path)

print(f"El modelo espera {model.n_features_in_} características.")

# 2. Solicitar datos del buyer desde la consola
print("Introduce los siguientes datos del comprador para calcular el score:")

total_transactions = int(input("Total de transacciones realizadas: "))
total_amount_spent = float(input("Monto total gastado: "))
on_time_payments = int(input("Pagos realizados a tiempo: "))
late_payments = int(input("Pagos realizados fuera de tiempo: "))
bnpl_used = int(input("¿Ha utilizado BNPL anteriormente? (1 = sí, 0 = no): "))
bnpl_default = int(input("¿Ha incumplido un crédito BNPL? (1 = sí, 0 = no): "))

# 3. Crear el array de características del comprador para pasarlo al modelo
buyer_features = np.array([[
    total_transactions,
    total_amount_spent,
    on_time_payments,
    late_payments,
    bnpl_used,
    bnpl_default
]])

# 4. Calcular la probabilidad de ser apto para BNPL
y_proba = model.predict_proba(buyer_features)[:, 1]  # Probabilidad de la clase positiva (apto)

# 5. Calcular el score basado en la probabilidad (normalización entre 300 y 850)
min_score = 300
max_score = 850

# Normalización de las probabilidades a un rango de score (300 a 850)
calculated_score = min_score + (y_proba[0] * (max_score - min_score))

# 6. Mostrar el score calculado
print(f"\nEl score calculado para el comprador es: {calculated_score:.2f}")