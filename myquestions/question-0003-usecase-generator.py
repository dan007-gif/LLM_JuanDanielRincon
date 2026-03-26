import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_simular_rendimientos():
    """
    Genera un caso de uso aleatorio para simular_rendimientos
    """

    # ----------------------------
    # 1. Dimensiones
    # ----------------------------
    n_rows = random.randint(10, 30)

    usuarios = [f"user_{i}" for i in range(10)]
    tipos = ["acciones", "bonos", "criptomonedas"]

    # ----------------------------
    # 2. Datos aleatorios
    # ----------------------------
    df = pd.DataFrame({
        "usuario": [random.choice(usuarios) for _ in range(n_rows)],
        "monto_invertido": np.random.randint(1000, 10000, size=n_rows),
        "rendimiento": np.random.uniform(-0.1, 0.2, size=n_rows),
        "tipo_activo": [random.choice(tipos) for _ in range(n_rows)]
    })

    n_simulaciones = random.randint(50, 150)

    # ----------------------------
    # 3. INPUT
    # ----------------------------
    input_data = {
        "df": df.copy(),
        "n_simulaciones": n_simulaciones
    }

    # ----------------------------
    # 4. OUTPUT esperado
    # ----------------------------
    np.random.seed(42)

    df_copy = df.copy()

    simulaciones_promedio = []

    for r in df_copy["rendimiento"]:
        sims = np.random.normal(loc=r, scale=0.05, size=n_simulaciones)
        simulaciones_promedio.append(sims.mean())

    df_copy["rendimiento_simulado_promedio"] = simulaciones_promedio

    resultado = (
        df_copy
        .groupby("tipo_activo")
        .agg({
            "rendimiento_simulado_promedio": ["mean", "std"]
        })
    )

    resultado.columns = ["promedio_rendimiento", "desviacion_rendimiento"]
    resultado = resultado.reset_index()

    output_data = resultado

    return input_data, output_data


# Ejemplo
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_simular_rendimientos()

    print("=== INPUT ===")
    print(entrada["df"].head())
    print("Simulaciones:", entrada["n_simulaciones"])

    print("\n=== OUTPUT ===")
    print(salida)