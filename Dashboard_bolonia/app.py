import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, confusion_matrix



img_icon = Image.open("./img/icon.png")
st.set_page_config(
    page_title="Dashboard Airbnb Bolonia",
    page_icon=img_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

Temas_pagina = {
    "Claro": {"primary": "#27445D", "bg": "#497D74"},#BF3131  #27445D
    "Oscuro": {"primary": "black", "bg": "#4F1C51"},
    "Personalizado": {"primary": "#F16F6F", "bg": "#9DC08B"}
}

with st.sidebar:
    selected = st.selectbox("Seleccionar tema", list(Temas_pagina.keys()))

    # Mostrar color pickers solo si se selecciona "Personalizado"
    if selected == "Personalizado":
        Temas_pagina["Personalizado"]["primary"] = st.color_picker(
            "Color principal", "#F16F6F")
        Temas_pagina["Personalizado"]["bg"] = st.color_picker(
            "Color de fondo", "#9DC08B")

theme = Temas_pagina[selected]


# Función para cargar datos con caché
@st.cache_data
def load_data():
    df = pd.read_csv("Data_clean_BO.csv")
    df = df.drop(['Unnamed: 0', 'Unnamed: 0.1','host_total_listings_count','number_of_reviews_ltm','number_of_reviews_l30d','id','host_id','scrape_id'], axis=1)


    #string a nuemrica
    #convertimos string a numéricos
    df['host_is_superhost'] = df['host_is_superhost'].replace({'f': 0, 't': 1})
    df['host_identity_verified'] = df['host_identity_verified'].replace({'f': 0, 't': 1}).astype(int)
    df['instant_bookable'] = df['instant_bookable'].replace({'f': 0, 't': 1}).astype(int)

    df['host_response_rate'] = df['host_response_rate'].astype(str).str.rstrip('%')
    df['host_response_rate'] = pd.to_numeric(df['host_response_rate'], errors='coerce').fillna(0)

    #Clasificación de columnas
    numeric_df = df.select_dtypes(include=['int','float','number'])
    numeric_cols = numeric_df.columns.tolist()

    text_df = df.select_dtypes(include=['object'])
    text_cols = text_df.columns.tolist()


    df['price'] = df['price'].astype(str).str.replace('$', '').str.replace(',', '')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    valor_euro = 18.50
    df["price_mx"] = df["price"] * valor_euro

    categorical_info = {
        'room_type': {
            'values': df['room_type'].unique().tolist(),
            'count': df['room_type'].nunique()
        },
    }

    return df, numeric_cols, text_cols, categorical_info, numeric_df

#####################################
################################################
###############DISEÑO DE LA INTERFAZ##############
################################################
#####################################
df, numeric_cols, text_cols, categorical_info, numeric_df = load_data()

st.markdown(f"""
<style>
    .stApp {{
        background-color: {theme['bg']};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {theme['primary']}; 
    }}
    h1, h2, h3 {{
        color: {theme['primary']};
        text-align: center;
        font-family: 'Roboto';
    }}
    body, td, th {{
    font-family: 'Segoe UI', Arial, sans-serif;
    }}
    .stApp header {{
        background-color: {theme['primary']};
        height: 0;
    }}
    .stImage > img {{
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# =============================================
# BARRA LATERAL
# =============================================
with st.sidebar:
    st.image("./img/icon.png", width=150)
    st.title("Filtros")

    # Filtro por tipo de habitación
    room_type = st.selectbox(
        "Tipo de Habitación",
        options=df['room_type'].unique().tolist(),
        index=0
    )

    # Filtro de rango de precios
    min_price, max_price = st.slider(
        "Rango de Precios (€)",
        min_value=int(df['price'].min()),
        max_value=int(df['price'].max()),
        value=(int(df['price'].quantile(0.25)), int(df['price'].quantile(0.75)))
    )

    # Filtro de rango de precios
    min_price_mx, max_price_mx = st.slider(
        "Rango de Precios (MX)",
        min_value=int(df['price_mx'].min()),
        max_value=int(df['price_mx'].max()),
        value=(int(df['price_mx'].quantile(0.25)), int(df['price_mx'].quantile(0.75)))
    )

    # Filtro de número mínimo de reviews
    min_reviews, max_reviews = st.slider(
        "Mínimo de Reviews",
        min_value=int(df['number_of_reviews'].min()),
        max_value=int(df['number_of_reviews'].max()),
        value=(int(df['number_of_reviews'].quantile(0.25)), int(df['number_of_reviews'].quantile(0.75)))
    )

# Aplicar filtros
filtered_df = df[
    (df['room_type'] == room_type) &
    (df['price'] >= min_price) &
    (df['price'] <= max_price) &
    (df['number_of_reviews'] >= min_reviews) &
    (df['number_of_reviews'] <= max_reviews) 
]
# Paleta de colores dinámica basada en room_type
def get_dynamic_colors(categories):
    base_color = theme['primary']  # Usa el color primario del tema
    colors = px.colors.qualitative.Plotly  #predefinida

    if len(categories) <= len(colors):
        return colors[:len(categories)]
    else:
        return px.colors.qualitative.Light24[:len(categories)]

##############################################
##############################################

# Menú de navegación superior
view = st.selectbox(
    "Seleccione una sección",
    options=["Inicio", "Análisis Exploratorio", "Modelado Predictivo"],
    key="nav_menu"
)

if view == "Inicio":
    # Header con columnas
    icon, titulo= st.columns([1, 4]) ###creacion de columnas
    with icon:
        st.image("./img/air.png", width=150)
    with titulo:
        st.title("Dashboard Airbnb Bolonia")
        st.markdown("Análisis de datos de alojamientos en Bolonia, Italia")

    # Datos clave
    prop, eur, mx, rev, ocu = st.columns(5)
    with prop:
        st.metric("Total Propiedades", len(filtered_df))
    with eur:
        st.metric("Precio Promedio", f"€{filtered_df['price'].mean():.2f}")
    with mx:
        st.metric("Precio Promedio", f"${filtered_df['price_mx'].mean():.2f}")
    with rev:
        st.metric("Reviews Promedio", f"{filtered_df['number_of_reviews'].mean():.1f}")
    with ocu:
        st.metric("Ocupación Promedio", f"{filtered_df['availability_365'].mean():.1f} días")


    # Mapa de ubicaciones
    st.subheader("Mapa de Alojamientos")
    st.map(filtered_df[['latitude', 'longitude']].dropna())
     # Uso en gráficos
    room_types = df['room_type'].unique()
    colors = get_dynamic_colors(room_types)

    fig = px.bar(df, x='room_type', y='price', color='room_type',
                    color_discrete_sequence=colors)
    st.plotly_chart(fig)

    #Seccion de texto
    st.subheader(" Acerca de Bolonia. ¿Por qué visitarlo?")
    tab1, tab2 = st.tabs(["Turismo", "Gastronomía"])
    with tab1:
        img1, img2, img3, img4 = st.columns([4,4,4,4])
        with img1:
            st.image("./img/porti.png", width=5060) 
        with img2: 
            st.image("./img/port.png", width=2500)
        with img3:
            st.image("./img/palace.png", width=4000) 
        with img4: 
            st.image("./img/bo3.png", width=4000)    
        st.write("""
        Bolonia, conocida como "La Docta" por su universidad (la más antigua de Europa occidental), 
        "La Roja" por el color de sus tejados y edificios, y "La Gorda" por su excelente gastronomía.
        """)
        st.write("""
        Principales lugares turísticos:
        - Pórticos de Bolonia
        - Palacio de Accursio
        - Piazza Maggiore
        - Fuente de Neptuno
        - Palacio Archiginnasio
        """)
    with tab2:
        img1, img2, img3, img4 = st.columns([4,4,4,4])
        with img1:
            st.image("./img/lasagna.png", width=5060) 
        with img2: 
            st.image("./img/bolon.png", width=2500)
        with img3:
            st.image("./img/tortellini.png", width=4000) 
        with img4: 
            st.image("./img/scar.png", width=4000) 
        st.write("""
        La cocina boloñesa es famosa mundialmente por platos como:
        - Tagliatelle al ragú (pasta a la boloñesa)
        - Lasagna alla bolognese
        - Tortellini en brodo
        - Mortadela
        """)

elif view == "Análisis Exploratorio":
    st.title("Análisis Exploratorio")

    # Distribución de precios
    st.subheader("Distribución de Precios")
    fig = px.histogram(
        filtered_df,
        x='price',
        nbins=50,
        title=f"Distribución de Precios para {room_type}",
        labels={'price': 'Precio (€)'},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Relación entre variables
    st.subheader("Relación entre Variables")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox(
            "Eje X",
            options=numeric_cols,
            index=0
        )
    with col2:
        y_axis = st.selectbox(
            "Eje Y",
            options=numeric_cols,
            index=1
        )

    fig = px.scatter(
        filtered_df,
        x=x_axis,
        y=y_axis,
        color='room_type',
        trendline="lowess",
        title=f"{y_axis} vs {x_axis}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparación entre tipos de habitación
    st.subheader("Comparación entre Tipos de Habitación")
    metric = st.selectbox(
        "Métrica a comparar",
        options=numeric_cols,
        index=0
    )

    fig = px.box(
        df,
        x='room_type',
        y=metric,
        title=f"Distribución de {metric} por tipo de Habitación"
    )
    st.plotly_chart(fig, use_container_width=True)

elif view == "Modelado Predictivo":
    st.title("Modelado Predictivo")

    model_type = st.radio(
        "Tipo de Modelo",
        options=["Regresión Lineal Simple", "Regresión Lineal Múltiple", "Regresión Logística"],
        horizontal=True
    )

    if model_type == "Regresión Lineal Simple":
        st.subheader("Regresión Lineal Simple")

        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox(
                "Variable Independiente (X)",
                options=numeric_cols,
                index=0
            )
        with col2:
            target = st.selectbox(
                "Variable Dependiente (y)",
                options=numeric_cols,
                index=0
            )

        X = filtered_df[[feature]]
        y = filtered_df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("R²: ", f"{r2:.4f}")
            st.write(f"Coeficiente: {model.coef_[0]:.4f}")
            st.write(f"Intercepto: {model.intercept_:.4f}")

        with col2:
            fig = px.scatter(
                x=X_test[feature],
                y=y_test,
                trendline="ols",
                trendline_color_override = "aqua",
                labels={'x': feature, 'y': target},
                title=f"{target} vs {feature}"
            )
            st.plotly_chart(fig, use_container_width=True)


    elif model_type == "Regresión Lineal Múltiple":
        st.subheader("Regresión Lineal Múltiple")

        features = st.multiselect(
            "Variables Independientes (X)",
            options=numeric_cols,
            default=['number_of_reviews', 'minimum_nights']
        )

        target = st.selectbox(
            "Variable Dependiente (y)",
            options=numeric_cols,
            index=0
        )

        if len(features) >= 2:
            X = filtered_df[features]
            y = filtered_df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train) #entrenamineto del modelo
            y_pred = model.predict(X_test) #prediccion

            r2 = r2_score(y_test, y_pred)

            st.metric("R² Score", f"{r2:.4f}")

            # Mostrar coeficientes
            coef_df = pd.DataFrame({
                'Variable': features,
                'Coeficiente': model.coef_
            })
            st.dataframe(coef_df)

            # Gráfico de residuos
            fig = px.scatter(
                x=y_pred,
                y=y_test - y_pred,
                labels={'x': 'Predicciones', 'y': 'Residuos'},
                title="Análisis de predicciones",
            )
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Seleccione al menos 2 variables predictoras")

    elif model_type == "Regresión Logística":
        st.subheader("Regresión Logística")

        # Crear variable binaria (precio alto/bajo)
        median_price = filtered_df['price'].median()
        filtered_df = filtered_df.copy()
        filtered_df['price_category'] = (filtered_df['price'] > median_price).astype(int)


        features = st.multiselect(
        "Variables Independientes (X)",
        options=numeric_cols,
        default=['number_of_reviews', 'minimum_nights']
            )

        if features:
            X = filtered_df[features]
            y = filtered_df['price_category']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = model.score(X_test, y_test)

            st.metric("Exactitud del Modelo", f"{accuracy:.2%}")

            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                labels=dict(x="Predicho", y="Real", color="Cantidad"),
                x=['Precio Bajo', 'Precio Alto'],
                y=['Precio Bajo', 'Precio Alto'],
                text_auto=True,
                title="Matriz de Confusión"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Coeficientes
            coef_df = pd.DataFrame({
                'Variable': features,
                'Coeficiente': model.coef_[0]
            })
            st.dataframe(coef_df)


