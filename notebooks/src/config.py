import os


# Caminho da pasta do projeto
PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
            )
        )
    )

# Caminho da pasta de dados
DATA_DIR = os.path.join(
    PROJECT_DIR, 
    "data"
    )

# Caminho da pasta de modelos
MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

# Caminho do modelo final
FINAL_MODEL = os.path.join(
    MODELS_DIR,
    "ridge_polyfeat_target_quantile.joblib"
)

# Caminho da base de dados
HOUSING_PATH = os.path.join(
    DATA_DIR,
    "housing.csv.zip"
)

# Caminho para a base tratada
CLEANED_PATH = os.path.join(
    DATA_DIR,
    "cleaned_housing.parquet",
)

# Caminho para o geojson da california original
GEO_CALIFORNIA_ORIGINAL_DIR = os.path.join(
    DATA_DIR,
    "california-counties.geojson"
)

# Caminho para a base com os dados dos condados agregados
AGG_COUNTIES_DIR = os.path.join(
    DATA_DIR,
    "counties.parquet"
)