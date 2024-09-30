import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import plotly.express as px
import dash
from dash import dcc, html

# 1. Cargar y preparar los datos
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    return df

# 3. Agregar reglas adicionales para etiquetar el riesgo
def label_risk(features):
    # Regla 1: Tasa de rechazo alta mayor al 0.10%
    rejection_rate_rule = (features['rejection_rate'] > 0.1).astype(int)
    
    # Regla 2: Identifica al 10% de los clientes con mayor volumen total de transacciones “altamente gastadores”
    high_total_spent_rule = (features['transaction_amount_volume'] > features['transaction_amount_volume'].quantile(0.9)).astype(int)
    
    # Regla 3: Pocas transacciones, pero con un monto promedio alto (alto riesgo)
    low_frequency_high_spending_rule = ((features['transaction_frequency'] < features['transaction_frequency'].quantile(0.2)) & 
                                        (features['avg_transaction_amount'] > features['avg_transaction_amount'].quantile(0.8))).astype(int)
    
    # Regla 4: Alta variabilidad en los montos de transacción (std alta)
    high_std_rule = (features['std_transaction_amount'] > features['std_transaction_amount'].quantile(0.9)).astype(int)
    
    # Regla 5: Tasa de disputas alta
    high_dispute_rate_rule = (features['dispute_rate'] > features['dispute_rate'].quantile(0.9)).astype(int)

    # Regla 6: Esta regla identifica a los clientes cuya tasa de transacciones en MCCs de alto riesgo está en el 20% superior.
    high_risk_mcc_rule = (features['high_risk_spending_rate'] > features['high_risk_spending_rate'].quantile(0.8)).astype(int)

    # Combinamos todas las reglas para crear una etiqueta final de riesgo
    features['is_risky'] = (rejection_rate_rule + 
                            high_total_spent_rule + 
                            low_frequency_high_spending_rule + 
                            high_std_rule +
                            high_dispute_rate_rule + 
                            high_risk_mcc_rule) > 2   # Al menos 2 reglas deben cumplirse
    
    return features['is_risky']

# 4. Entrenar el modelo
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba)}")

    joblib.dump(model, 'output/credit_risk_model.pkl')
    joblib.dump(scaler, 'output/credit_risk_scaler.pkl')
    return model, scaler

def createDashboard(model, X, df):
    # Extraer los coeficientes del modelo
    feature_importance = abs(model.coef_[0])
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values(by='importance', ascending=False)

    # 5. Crear el dashboard con Dash
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Dashboard de Riesgo"),
        
        # Gráfico de distribución de etiquetado
        dcc.Graph(id='risk-distribution', figure=px.histogram(df, x='is_risky', title="Distribución de Etiquetas de Riesgo")),
        
        # Gráfico de importancia de características
        dcc.Graph(id='feature-importance', figure=px.bar(importance_df, x='feature', y='importance', 
                                                        title="Importancia de las Características (Coeficientes Absolutos)")),
    ])
    app.run_server(debug=True)

# 5. Función principal
def main(run_dashboard=False):
    # Leer archivo con las feaures calculas del hitorico de transacciones por customer
    df = load_and_prepare_data('data/customer_features.csv')
    
    # Definir y ordenar los campos específicos que deseas utilizar en X
    selected_columns = [
        'transaction_amount_volume', 
        'avg_transaction_amount',
        'std_transaction_amount', 
        'days_since_first_transaction',
        'transaction_count',
        'transaction_frequency',
        'rejection_rate', 
        'dispute_rate', 
        'high_risk_spending_rate'
        ]
    
    # Seleccionar solo esas columnas de X
    X = df[selected_columns]

    # Aplicar las reglas para etiquetar a los buyers como 'riesgosos' o 'no riesgosos'
    y = label_risk(df)

    # Entrenar modelo
    model, scaler = train_model(X, y)

    # Visualizar dashboard de features y etiquetas
    if run_dashboard:
        createDashboard(model, X, df)

if __name__ == "__main__":
    main(run_dashboard=False)