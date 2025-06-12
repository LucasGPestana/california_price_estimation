import streamlit as st
import pandas as pd
import geopandas as gpd


import joblib


from notebooks.src.config import (
    AGG_COUNTIES_DIR, 
    CLEANED_PATH, 
    FINAL_MODEL
)


@st.cache_data
def loadCleanData():

    return pd.read_parquet(CLEANED_PATH)

@st.cache_data
def loadGeoData():

    return pd.read_parquet(AGG_COUNTIES_DIR)

@st.cache_resource
def loadModel():

    return joblib.load(FINAL_MODEL)

df = loadCleanData()
gdf_geo = loadGeoData()
model = loadModel()


st.title('Previsão de preços de imóveis')

longitude = st.number_input("Longitude", value=-122.33)
latitude = st.number_input("Latitude", value=37.33)

housing_median_age = st.number_input("Idade do imóvel", value=10)

total_rooms = st.number_input("Total de cômodos", value=100)
total_bedrooms = st.number_input("Total de quartos", value=100)
population = st.number_input("População", value=100)
households = st.number_input("Domicilios", value=100)

median_income = st.slider(
    "Renda anual média (múltipos de US$ 10k)", 
    0.5, 15.0, 
    value=4.5, 
    step=0.5
    )

ocean_proximity = st.selectbox("Proximidade do oceano", df["ocean_proximity"].unique())

rooms_per_households = st.number_input("Cômodos por domicilio", value=7)
population_per_households = st.number_input("População por domicilio", value=0.2)
bedrooms_per_rooms = st.number_input("Quartos por cômodo", value=2)

median_income_cat = st.number_input("Categoria de renda", value=4)
housing_median_age_cat = st.number_input("Categoria de idade do imóvel", value=4)

model_input = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean_proximity,
    "rooms_per_households": rooms_per_households,
    "population_per_households": population_per_households,
    "bedrooms_per_rooms": bedrooms_per_rooms,
    "median_income_cat": median_income_cat,
    "housing_median_age_cat": housing_median_age_cat,
}

df_model_input = pd.DataFrame(model_input, index=[0])

button_state = st.button("Prever preço")

if button_state:

    price = model.predict(df_model_input)

    st.write(f"Preço: {price[0]:.2f}")