## **Descripción del Proyecto**

El proyecto es un ejercicio practico de la epsecializacion en Data Science G8 de Alura y  el objetivo principal es **desarrollar un modelo de regresión para predecir los retrasos en vuelos**. Esto implica analizar un conjunto de datos históricos de vuelos para identificar factores que influyen en los retrasos y construir un modelo predictivo capaz de estimar la duración de dichos retrasos. El análisis y modelado buscan contribuir a la **optimización de las operaciones aeroportuarias** [previsionVuelos/optimizacion_aeroportuaria.ipynb].

## **Tecnologías y Librerías Utilizadas**

El proyecto utiliza un conjunto de librerías de Python para el análisis de datos y el aprendizaje automático:

*   **Pandas** (`pd`): Versión 2.2.2 - Para la manipulación y análisis de datos.
*   **NumPy** (`np`): Versión 2.0.2 - Para operaciones numéricas eficientes.
*   **Scikit-learn** (`sklearn`): Versión 1.6.1 - Para la creación y evaluación de modelos de aprendizaje automático.
*   **Seaborn** (`sns`): Versión 0.13.2 - Para la visualización estadística de datos.
*   **Yellowbrick**: Versión 1.5 - Para la visualización de diagnósticos de modelos de *machine learning*, como errores de predicción y análisis de residuos.
*   **Matplotlib** (`plt`): Utilizado para la creación de gráficos y visualizaciones.

## **Conjunto de Datos**

El análisis se realiza sobre un conjunto de datos cargado desde un archivo `flights.csv`.

<img width="637" height="352" alt="Captura" src="https://github.com/user-attachments/assets/e4703d71-277e-4a5d-824c-97e61d4a3c60" />

## **Análisis Exploratorio de Datos (EDA)**

Se realizaron varias visualizaciones y análisis para entender las características de los datos:

*   **Atraso Promedio por Categoría**: Gráficos de barras que muestran el retraso promedio en función de la aerolínea, el tipo de vuelo (Schengen/No-Schengen) y si el día es festivo.
*   **Número de Vuelos por Categoría**: Gráficos de conteo para la distribución de vuelos por aerolínea, tipo de vuelo y tipo de aeronave.
*   **Distribución de Tiempos**: Histogramas para `arrival_time` y `departure_time`, utilizando la regla de Freedman Diaconis para el ancho de los bins.
*   **Distribución de Retrasos**: Boxplots e histogramas para la variable `delay`, mostrando su media y mediana. Se observa que el retraso tiene una media de aproximadamente 12.55 minutos y una mediana de 9.74 minutos.
<img width="700" height="507" alt="image" src="https://github.com/user-attachments/assets/95338aa8-c086-4be3-83c0-91acd2d6575a" />

## **Preprocesamiento de Datos**

Para preparar los datos para el modelado, se llevaron a cabo los siguientes pasos:

*   **Creación de Nuevas Características**:
    *   Se generó una columna `date` combinando `year` y `day`.
    *   Se crearon las columnas `is_weekend` (indicador de fin de semana) y `day_name` (nombre del día de la semana) a partir de la columna `date`. El dataset resultante contiene 14 columnas.
*   **Codificación de Variables Categóricas**:
    *   Variables booleanas (`is_holiday`, `is_weekend`) y categóricas binarias (`schengen`) se transformaron a valores numéricos (0/1).
    *   Variables categóricas nominales como `airline`, `aircraft_type`, `day_name` y `origin` fueron convertidas usando **One-Hot Encoding** (`pd.get_dummies`), resultando en un dataframe con 36 columnas.
*   **Correlación**: Se encontró una correlación muy fuerte entre `arrival_time` y `departure_time` (0.973797).
*   **Selección de Características para el Modelo**: Se eliminaron columnas como `flight_id`, `year`, `day`, `date` y `departure_time` del conjunto de datos para el modelado, resultando en un conjunto final de **31 características**.

## **Modelado y Evaluación**

El proyecto se enfoca en un **modelo de regresión** para predecir `delay`.

*   **División de Datos**: El conjunto de datos se dividió en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba) con un `random_state` de 50.
*   **Métricas de Evaluación**: Se utilizan las siguientes métricas de regresión para evaluar el rendimiento de los modelos:
    *   **MSE** (Mean Squared Error) - Específicamente, el **RMSE** (Raíz Cuadrada del Error Cuadrático Medio) para interpretar los errores en las mismas unidades que la variable objetivo.
    *   **MAE** (Mean Absolute Error) - Error Absoluto Medio.
    *   **R2** (Coeficiente de Determinación) - Proporción de la varianza en la variable dependiente que es predecible a partir de las variables independientes.

### **Modelos Implementados**

1.  **Modelo Baseline (DummyRegressor)**:
    *   Un modelo simple que predice la media de la variable objetivo.
    *   **Resultados Baseline**: RMSE: 22.941, MAE: 18.4462, R2: -0.0003. Estos resultados sirven como punto de referencia para modelos más complejos.

2.  **RandomForestRegressor**:
    *   Se entrenó un modelo de **Random Forest Regressor** con una profundidad máxima (`max_depth`) de 5 y `random_state` de 50.
    *   **Resultados del Modelo**: **RMSE: 13.7014, MAE: 11.0399, R2: 0.6432**. Esto indica una mejora significativa sobre el modelo baseline.

### **Diagnóstico del Modelo**

Se utilizaron visualizadores de Yellowbrick para el diagnóstico del modelo de Random Forest:

*   **PredictionError**: Muestra el error de predicción del modelo.
*   **ResidualsPlot**: Analiza los residuos del modelo, verificando si tienen un comportamiento similar a los datos de entrenamiento.

<img width="504" height="509" alt="image" src="https://github.com/user-attachments/assets/f0711ef8-17d9-4b1d-9358-1ecf2e765cbc" />

### **Validación Cruzada**

Se realizó una **validación cruzada de 5 pliegues (K-Fold)** con `shuffle=True` y `random_state=50` para evaluar la robustez del modelo. Las métricas evaluadas fueron el error cuadrático medio negativo, el error absoluto medio negativo y el R2.

### **Importancia de las Características**

Se analizó la importancia de las características para el modelo Random Forest. Las características más importantes incluyen:

*   **`airline_BZ`**: 52.88% de importancia.
*   **`is_holiday`**: 15.01% de importancia.
*   **`aircraft_type_Airbus A320`**: 9.75% de importancia.

Se exploró cómo el rendimiento del modelo cambia al incluir un número creciente de las características más importantes, observándose la estabilidad de las métricas (`mse`, `mae`, `r2`) después de cierto punto.

## **Uso y ejecución del proyecto**
* Clona este repositorio.
* Abre el archivo optimizacion_aeroportuaria.ipynb en Jupyter Notebook o Google Colab.
* Ejecuta las celdas para reproducir la predicción del modelado

## **Estructura del Repositorio**

```
previsionVuelos/
├── optimizacion_aeroportuaria.ipynb   # Notebook principal del proyecto
├── flights.csv                          # Dataset utilizado
├── README.md                            # Este archivo
└── [LICENSE]                            # [Archivo de licencia]
```


## **Licencia**
Este proyecto está licenciado bajo la Licencia MIT.

Autor: Ana Sayago
