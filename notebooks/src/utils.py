import pandas as pd
import numpy as np


from typing import Sequence

def organizeResults(results) -> pd.DataFrame:

    df = pd.DataFrame(
        results
    )

    df = df.T

    # Verifica se há as colunas fit_time e score_time
    if len(df.columns.difference(["fit_time", "score_time"])) < len(df.columns):

        df["time"] = df["fit_time"] + df["score_time"]
        df = df.drop(["fit_time", "score_time"], axis=1)

    df.index = df.index.rename("model")

    df = df.explode(
        df.columns.tolist()
        ).reset_index()

    return df

def getCoeffsDataframe(coeffs: np.ndarray, feature_names: Sequence[str]) -> pd.DataFrame:

    if coeffs.ndim == 1:

        coeffs = coeffs.reshape(-1, 1)

    df = pd.DataFrame(
        coeffs, 
        columns=["coeficientes"], 
        index=feature_names
        ).sort_values("coeficientes", ascending=True)
    
    return df