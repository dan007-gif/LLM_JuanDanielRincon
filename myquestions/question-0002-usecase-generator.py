import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analizar_canciones():
    """
    Genera un caso de uso aleatorio para analizar_canciones
    """

    # ----------------------------
    # 1. Dimensiones
    # ----------------------------
    n_rows = random.randint(20, 50)

    generos = ["rock", "pop", "reggaeton", "jazz", "electronic"]
    artistas = ["artista_" + str(i) for i in range(10)]

    # ----------------------------
    # 2. Datos aleatorios
    # ----------------------------
    df = pd.DataFrame({
        "artista": [random.choice(artistas) for _ in range(n_rows)],
        "genero": [random.choice(generos) for _ in range(n_rows)],
        "reproducciones": np.random.randint(1000, 1000000, size=n_rows),
        "duracion_segundos": np.random.randint(120, 400, size=n_rows),
        "popularidad": np.random.randint(1, 101, size=n_rows)
    })

    # ----------------------------
    # 3. INPUT
    # ----------------------------
    input_data = {
        "df": df.copy()
    }

    # ----------------------------
    # 4. OUTPUT esperado
    # ----------------------------
    df_copy = df.copy()

    df_copy["eficiencia_popularidad"] = df_copy["popularidad"] / df_copy["duracion_segundos"]

    promedio_global = df_copy["reproducciones"].mean()

    filtrado = df_copy[df_copy["reproducciones"] > promedio_global]

    resultado = (
        filtrado
        .groupby("genero")
        .agg({
            "eficiencia_popularidad": "mean",
            "popularidad": "var",
            "reproducciones": lambda x: np.percentile(x, 75)
        })
        .rename(columns={
            "eficiencia_popularidad": "promedio_eficiencia",
            "popularidad": "varianza_popularidad",
            "reproducciones": "p75_reproducciones"
        })
        .reset_index()
        .sort_values(by="p75_reproducciones", ascending=False)
        .reset_index(drop=True)
    )

    output_data = resultado

    return input_data, output_data


# ----------------------------------
# EJEMPLO 
# ----------------------------------
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_analizar_canciones()

    print("=== INPUT (Diccionario) ===")
    print("DataFrame (primeras filas):")
    print(entrada["df"].head())

    print("\n=== OUTPUT ESPERADO ===")
    print(salida)