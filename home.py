import streamlit as st
import pydeck as pdk
import pandas as pd
import geopandas as gpd
import shapely
import numpy as np
import joblib


from typing import Any, Sequence, List, Tuple
import math


from notebooks.src.config import (
    AGG_COUNTIES_DIR, 
    CLEANED_PATH, 
    FINAL_MODEL
)


@st.cache_data
def loadCleanData() -> pd.DataFrame:

    return pd.read_parquet(CLEANED_PATH)

@st.cache_data
def loadGeoData() -> gpd.GeoDataFrame:

    gdf_geo = gpd.read_parquet(AGG_COUNTIES_DIR)

    # 'Explodindo' os Multipolygon em Polygon individuais
    gdf_geo = gdf_geo.explode(ignore_index=True)

    def fixGeometry(geometry: shapely.Geometry) -> shapely.Geometry:

        """Conserta as geometrias inválidas

        Parameters
        ----------
        geometry : shapely.Geometry
            Geometria inválida a ser consertada
        
        Returns
        -------
        shapely.Geometry
            Geometria consertada
        
        """

        if geometry.is_valid:

            geometry = geometry.buffer(0)
        
        # Orienta a geometria no sentido horário caso seja um Polygon ou MultiPolygon
        if isinstance(geometry, (shapely.Polygon, shapely.MultiPolygon)):

            geometry = shapely.geometry.polygon.orient(geometry, sign=1.0)
        
        return geometry
    
    gdf_geo["geometry"] = gdf_geo["geometry"].apply(fixGeometry)

    def getPolygonCoordinates(geometry: shapely.Geometry) -> List[List[Tuple[float, float]]]:

        """Obtém as coordenadas de uma geometria

        Parameters
        ----------
        geometry : shapely.Geometry
            Geometria a ser extraida as coordenadas
        
        Returns
        -------
        List[List[Tuple[float, float]]]
            Lista com as coordenadas
        
        """

        if isinstance(geometry, shapely.Polygon):

            coords = [[(x, y) for x, y in geometry.exterior.coords]]
        
        else:

            coords = [[(x, y) for x, y in geometry.exterior.coords] for geometry in geometry.geoms]
        
        return coords

    gdf_geo["geometry"] = gdf_geo["geometry"].apply(getPolygonCoordinates)

    return gdf_geo

@st.cache_resource
def loadModel():

    return joblib.load(FINAL_MODEL)

def getCategoryValue(value: float, limits: Sequence[float], labels: Sequence[Any]) -> Any:

    cat_idx = np.digitize(
        value, 
        bins=limits, 
        right=True
    )

    return labels[cat_idx - 1]


df = loadCleanData()
gdf_geo = loadGeoData()
model = loadModel()

st.title('Previsão de preços de imóveis')

counties = sorted(gdf_geo["name"].unique())

col1, col2 = st.columns(2)

with col1:

    with st.form("form"):

        selected_county = st.selectbox("Condado", counties)

        longitude = gdf_geo.query("name == @selected_county")["longitude"].values[0]
        latitude = gdf_geo.query("name == @selected_county")["latitude"].values[0]

        housing_median_age = st.number_input(
            "Idade do imóvel", 
            value=10, 
            min_value=df["housing_median_age"].min(), 
            max_value=df["housing_median_age"].max()
            )

        total_rooms = gdf_geo.query("name == @selected_county")["total_rooms"].values[0]
        total_bedrooms = gdf_geo.query("name == @selected_county")["total_bedrooms"].values[0]
        population = gdf_geo.query("name == @selected_county")["population"].values[0]
        households = gdf_geo.query("name == @selected_county")["households"].values[0]

        median_income = st.slider(
            "Renda anual média, em US$", 
            min_value=5_000.00, 
            max_value=90_000.00,
            value=45_000.00, 
            step=0.01
            ) / 10_000

        ocean_proximity = gdf_geo.query("name == @selected_county")["ocean_proximity"].values[0]

        rooms_per_households = gdf_geo.query("name == @selected_county")["rooms_per_households"].values[0]
        population_per_households = gdf_geo.query("name == @selected_county")["population_per_households"].values[0]
        bedrooms_per_rooms = gdf_geo.query("name == @selected_county")["bedrooms_per_rooms"].values[0]

        median_income_cat_limits = np.append(np.arange(0, 7, 1.5), np.inf)
        median_income_cat_labels = list(range(1, len(median_income_cat_limits)))

        median_income_cat = getCategoryValue(
            median_income,
            limits=median_income_cat_limits,
            labels=median_income_cat_labels
        )


        housing_median_age_cat_limits = [0, 18, 30, 50, np.inf]
        housing_median_age_cat_labels = list(range(1, len(housing_median_age_cat_limits)))

        housing_median_age_cat = getCategoryValue(
            housing_median_age,
            limits=housing_median_age_cat_limits,
            labels=housing_median_age_cat_labels
        )

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

        button_state = st.form_submit_button("Prever preço")

    if button_state:

        price = model.predict(df_model_input)

        st.metric(
            label="Preço estimado",
            value=f"US$ {str(round(price[0], 2)).replace('.', ',')}"
            )

with col2:

    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=5,
        min_zoom=5,
        max_zoom=15,
    )

    polygon_layer = pdk.Layer(
        "PolygonLayer",
        data=gdf_geo[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[0, 0, 255, 100],
        get_line_color=[255, 255, 255],
        get_line_width=50,
        pickable=True,
        auto_highlight=True,
    )

    county_data = gdf_geo.query("name == @selected_county")

    highlight_layer = pdk.Layer(
        "PolygonLayer",
        data=county_data[["name", "geometry"]],
        get_polygon="geometry",
        get_fill_color=[255, 0, 0, 100],
        get_line_color=[0, 0, 0],
        get_line_width=500,
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": "<strong>Condado: </strong><span>{name}</span>",
        "style": {"backgroundColor": "gray", "color": "white", "fontsize": "10px"}
    }

    map_obj = pdk.Deck(
        initial_view_state=view_state,
        layers=[polygon_layer, highlight_layer],
        map_style="light",
        tooltip=tooltip
    )

    st.pydeck_chart(
        map_obj,
        use_container_width=True,
        height=500,
        )
