import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score

def calcular_pureza(y_true, y_pred):
    """
    Calcula la pureza del clustering
    """
    df = pd.DataFrame({"true": y_true, "cluster": y_pred})
    total = len(df)

    pureza = 0
    for cluster in df["cluster"].unique():
        subset = df[df["cluster"] == cluster]
        max_count = subset["true"].value_counts().max()
        pureza += max_count

    return pureza / total


def generar_caso_de_uso_agrupar_usuarios():
    """
    Genera un caso de uso aleatorio para agrupar_usuarios
    """

    # ----------------------------
    # 1. Dimensiones
    # ----------------------------
    n_rows = random.randint(30, 60)

    tipos_usuario = ["casual", "frecuente", "intensivo"]

    # ----------------------------
    # 2. Datos aleatorios
    # ----------------------------
    df = pd.DataFrame({
        "tiempo_uso_diario": np.random.randint(10, 300, size=n_rows).astype(float),
        "numero_interacciones": np.random.randint(1, 100, size=n_rows).astype(float),
        "contenido_consumido": np.random.randint(1, 200, size=n_rows).astype(float),
        "nivel_actividad": np.random.uniform(0, 100, size=n_rows),
        "tipo_usuario": [random.choice(tipos_usuario) for _ in range(n_rows)]
    })

    # ----------------------------
    # 3. Introducir NaNs
    # ----------------------------
    cols_numericas = df.columns.drop("tipo_usuario")

    mask = np.random.choice(
        [True, False],
        size=df[cols_numericas].shape,
        p=[0.1, 0.9]
    )

    df.loc[:, cols_numericas] = df.loc[:, cols_numericas].mask(mask)

    n_clusters = random.randint(2, 4)

    # ----------------------------
    # 4. INPUT
    # ----------------------------
    input_data = {
        "df": df.copy(),
        "n_clusters": n_clusters
    }

    # ----------------------------
    # 5. OUTPUT esperado
    # ----------------------------
    df_copy = df.copy()

    y_true = df_copy["tipo_usuario"]

    X = df_copy[cols_numericas]

    imputer = SimpleImputer(strategy="constant", fill_value=0)
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(X_scaled)

    df_copy["cluster"] = clusters

    # métricas
    silhouette = silhouette_score(X_scaled, clusters)
    pureza = calcular_pureza(y_true, clusters)

    output_data = (df_copy, silhouette, pureza)

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_agrupar_usuarios()

    print("=== INPUT ===")
    print(entrada["df"].head())
    print("Clusters:", entrada["n_clusters"])

    print("\n=== OUTPUT ===")
    df_res, sil, pur = salida
    print(df_res.head())
    print("Silhouette:", sil)
    print("Pureza:", pur)