import joblib
import pandas as pd

def load_model_and_scaler():
    model = joblib.load('output/credit_risk_model.pkl')
    scaler = joblib.load('output/credit_risk_scaler.pkl')
    return model, scaler

def predict_risk(model, scaler, features):
    # Asegurar que las características estén en el mismo orden que durante el entrenamiento
    feature_order = [
        'transaction_amount_volume', 
        'avg_transaction_amount',
        'std_transaction_amount', 
        'days_since_first_transaction',
        'transaction_count',
        'transaction_frequency',
        'rejection_rate', 
        'dispute_rate', 
        'high_risk_spending_rate',
        ]
    
    # Crear un DataFrame con las características en el orden correcto
    feature_df = pd.DataFrame([features], columns=feature_order)
    
    # Escalar las características
    scaled_features = scaler.transform(feature_df)
    
    # Predecir la probabilidad
    risk_probability = model.predict_proba(scaled_features)[0][1]
    
    return risk_probability

def main():
    model, scaler = load_model_and_scaler()
    
    print("Ingrese los siguientes datos para el cliente:")
    features = {
        'transaction_amount_volume': float(input("Volumen total de transacciones: ")),
        'avg_transaction_amount': float(input("Monto promedio de transacción: ")),
        'std_transaction_amount': float(input("Desviación estándar de montos de transacción: ")),
        'days_since_first_transaction': int(input("Días desde la primera transacción: ")),
        'transaction_count': int(input("Número de transacciones: ")),
        'transaction_frequency': float(input("Frecuencia de transacciones (transacciones por día): ")),
        'rejection_rate': float(input("Tasa de rechazo (0-1): ")),
        'dispute_rate': float(input("Tasa de disputas (0-1): ")),
        'high_risk_spending_rate': float(input("Tasa de trxs con mcc riesgoso (0-1): "))
    }
    
    risk_probability = predict_risk(model, scaler, features)
    print(f"\nLa probabilidad de riesgo para este cliente es: {risk_probability:.2%}")

if __name__ == "__main__":
    main()