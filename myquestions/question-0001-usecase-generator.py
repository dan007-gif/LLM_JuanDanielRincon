import pandas as pd
import numpy as np
import random
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression

def generar_caso_de_uso_preparar_jugadores():
    """
    Genera un caso de uso aleatorio para la función preparar_jugadores.
    Devuelve un input (diccionario) y un output esperado (modelo entrenado y predicciones).
    """

    # 1. Configurar dimensiones aleatorias
    n_rows = random.randint(10, 30)  # Número de filas entre 10 y 30
    n_features = random.randint(3, 6)  # Número de características entre 3 y 6
    feature_cols = [f'feature_{i}' for i in range(n_features)]

    # 2. Generar datos aleatorios (con dispersión)
    data = np.random.randn(n_rows, n_features) * 100  # Amplitud para simular outliers
    df = pd.DataFrame(data, columns=feature_cols)

    # Introducir NaNs aleatorios (~15%)
    mask = np.random.choice([True, False], size=df.shape, p=[0.15, 0.85])
    df[mask] = np.nan

    # 3. Variable objetivo desbalanceada (20% positivos)
    target_col = "desafio_completado"
    y = np.random.choice([0, 1], size=n_rows, p=[0.8, 0.2])  # 80% 0, 20% 1
    df[target_col] = y

    # 4. Construir el input (diccionario de argumentos)
    input_data = {
        "df": df.copy(),  # Copia para no modificar el original
        "target_col": target_col
    }

    # 5. Calcular el output esperado (modelo entrenado y predicciones)
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputar valores faltantes
    imputer = KNNImputer()
    X_imputed = imputer.fit_transform(X)

    # Escalar con RobustScaler (manejar outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Entrenar modelo con class_weight='balanced'
    model = LogisticRegression(class_weight='balanced', max_iter=200)
    model.fit(X_scaled, y)

    # Predecir en X escalado
    y_pred = model.predict(X_scaled)

    # Output: modelo y predicciones
    output_data = (model, y_pred)

    return input_data, output_data

# Ejemplo de uso
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_preparar_jugadores()

    print("=== INPUT ===")
    print("Target:", entrada["target_col"])
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    modelo, predicciones = salida
    print("Modelo entrenado:", modelo)
    print("Predicciones (primeras 10):", predicciones[:10])