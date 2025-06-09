import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# --------------------------------------------
# Función para REalizar Análisis de cosnsistencia temporal
# --------------------------------------------


def validate_temporal_consistency(df):
    """
    Valida que no haya vacíos temporales en los datos de cada barrio.
    Usa las columnas 'id_bar', 'anio' y 'semana'.
    """
    print("Ejecutando validación de consistencia temporal...")
    df_sorted = df[['id_bar', 'anio', 'semana']].sort_values(['id_bar', 'anio', 'semana']).drop_duplicates()
    
    # Crear una columna de fecha para hacer los cálculos
    df_sorted['fecha'] = pd.to_datetime(df_sorted['anio'].astype(str) + df_sorted['semana'].astype(str).str.zfill(2) + '1', format='%Y%W%w')

    gaps = {}
    for barrio in df_sorted['id_bar'].unique():
        barrio_data = df_sorted[df_sorted['id_bar'] == barrio]
        start_date = barrio_data['fecha'].min()
        end_date = barrio_data['fecha'].max()
        
        # Crear todas las semanas esperadas en el rango de fechas del barrio
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='W-MON')

        # Comparar las semanas reales con las esperadas para encontrar las que faltan
        missing_dates = full_date_range.difference(barrio_data['fecha'])

        if not missing_dates.empty:
            gaps[barrio] = sorted([(date.year, date.isocalendar().week) for date in missing_dates])
        else:
            gaps[barrio] = []
    
    print("Validación completada.")
    return gaps


# --------------------------------------------
# Función para identificar valores atipicos
# --------------------------------------------

def detect_outliers_dengue(df, column_name='dengue'): # AJUSTADO: 'dengue' es ahora el valor por defecto
    """
    Detecta picos anómalos en los casos de dengue usando IQR y Z-score.
    """
    print(f"Ejecutando detección de outliers en la columna '{column_name}'...")
    df_outliers = df.copy()
    df_outliers['outlier_iqr'] = False
    df_outliers['outlier_zscore'] = False

    for barrio in df_outliers['id_bar'].unique():
        barrio_mask = df_outliers['id_bar'] == barrio
        barrio_data = df_outliers.loc[barrio_mask, column_name]

        # Método IQR
        Q1 = barrio_data.quantile(0.25)
        Q3 = barrio_data.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound_iqr = Q3 + 3 * IQR
        is_outlier_iqr = barrio_data > upper_bound_iqr
        df_outliers.loc[barrio_mask & is_outlier_iqr.reindex(df_outliers.index).fillna(False), 'outlier_iqr'] = True

        # Método Z-score
        mean = barrio_data.mean()
        std = barrio_data.std()
        if std > 0:
            z_scores = np.abs((barrio_data - mean) / std)
            is_outlier_zscore = z_scores > 3
            df_outliers.loc[barrio_mask & is_outlier_zscore.reindex(df_outliers.index).fillna(False), 'outlier_zscore'] = True
    
    print("Detección de outliers completada.")
    return df_outliers


# --------------------------------------------
# Función para Análisis de descomposición estacional 
# --------------------------------------------

def seasonal_decomposition_analysis(df, column_name='dengue', period=52):
    """
    Realiza y grafica una descomposición estacional para cada barrio.
    VERSIÓN CORREGIDA: Maneja fechas duplicadas sumando sus valores.
    """
    print(f"Ejecutando análisis de descomposición estacional en la columna '{column_name}'...")
    df_temp = df.copy()
    df_temp['fecha'] = pd.to_datetime(df_temp['anio'].astype(str) + df_temp['semana'].astype(str).str.zfill(2) + '1', format='%Y%W%w')
    df_temp = df_temp.set_index('fecha').sort_index()

    decompositions = {}
    for barrio in df_temp['id_bar'].unique():
        print(f"--- Procesando Barrio {barrio} ---")
        barrio_ts = df_temp[df_temp['id_bar'] == barrio][column_name]
        
        # --- LÍNEA CLAVE DE LA CORRECCIÓN ---
        # Si hay fechas duplicadas, las agrupamos y sumamos sus valores.
        if barrio_ts.index.has_duplicates:
            print(f"AVISO: Se encontraron fechas duplicadas para el barrio {barrio}. Sumando los valores.")
            barrio_ts = barrio_ts.groupby(level=0).sum()
        
        # Rellenar fechas faltantes para que el análisis funcione
        barrio_ts = barrio_ts.asfreq('W-MON').fillna(method='ffill')

        if len(barrio_ts) >= 2 * period:
            result = seasonal_decompose(barrio_ts, model='additive', period=period)
            decompositions[barrio] = result
            
            print(f"Graficando descomposición para el Barrio {barrio}...")
            fig = result.plot()
            fig.set_size_inches(12, 8)
            plt.suptitle(f'Descomposición Estacional para el Barrio {barrio}', y=0.95)
            plt.show()
        else:
            print(f"AVISO: No hay suficientes datos para el barrio {barrio} para realizar la descomposición.")
            print(f"(Se necesitan al menos {2 * period} semanas y solo hay {len(barrio_ts)}).")
            decompositions[barrio] = None
            
    print("Análisis de descomposición completado.")
    return decompositions

# --------------------------------------------
# Función para Identificar Picos Epidémicos
# --------------------------------------------

def identify_epidemic_peaks(df, dengue_col='dengue', threshold_percentile=90):
    """
    Identifica picos epidémicos, su duración y los intervalos entre ellos para cada barrio.
    VERSIÓN CORREGIDA: Convierte año/semana a un índice de fecha para calcular intervalos correctamente.
    """
    df_temp = df.copy()
    
    # --- PASO 1: CREAR Y ESTABLECER UN ÍNDICE DE FECHA (LA CORRECCIÓN CLAVE) ---
    try:
        df_temp['fecha'] = pd.to_datetime(df_temp['anio'].astype(str) + df_temp['semana'].astype(str).str.zfill(2) + '1', format='%Y%W%w')
        df_temp = df_temp.set_index('fecha').sort_index()
    except Exception as e:
        print(f"Error al convertir a fecha: {e}. Asegúrate que las columnas 'anio' y 'semana' son correctas.")
        return {}

    resultados = {}

    for barrio in df_temp['id_bar'].unique():
        barrio_data = df_temp[df_temp['id_bar'] == barrio].sort_index()
        
        # 2. Definir umbral del pico
        threshold = barrio_data[dengue_col].quantile(threshold_percentile / 100)
        
        # 3. Identificar semanas en pico
        barrio_data['en_pico'] = barrio_data[dengue_col] > threshold
        
        if not barrio_data['en_pico'].any():
            resultados[barrio] = {'duracion_picos': [], 'intervalo_entre_picos': []}
            continue

        # 4. Calcular duración e intervalos usando el índice de fecha
        pico_grupos = (barrio_data['en_pico'] != barrio_data['en_pico'].shift()).cumsum()
        
        duraciones = []
        intervalos = []
        fin_pico_anterior_fecha = None

        for group_id in pico_grupos.unique():
            grupo = barrio_data[pico_grupos == group_id]
            es_pico = grupo['en_pico'].iloc[0]

            if es_pico:
                # Es un pico, calculamos su duración
                duracion = len(grupo)
                duraciones.append(duracion)
                
                # Si hubo un pico anterior, calculamos el intervalo usando las fechas
                if fin_pico_anterior_fecha is not None:
                    inicio_pico_actual_fecha = grupo.index[0]
                    # La diferencia ahora es un Timedelta, calculamos las semanas
                    intervalo_semanas = (inicio_pico_actual_fecha - fin_pico_anterior_fecha).days // 7
                    intervalos.append(intervalo_semanas)
            else:
                # No es un pico, guardamos la fecha de fin de este "valle"
                # para el siguiente cálculo de intervalo.
                fin_pico_anterior_fecha = grupo.index[-1]
                
        resultados[barrio] = {'duracion_picos': duraciones, 'intervalo_entre_picos': intervalos}
        
    return resultados


# --------------------------------------------
# Función para Identificar Correlación Climática con Retraso (Lag)
# --------------------------------------------

def climate_dengue_correlation(df, dengue_col='dengue', max_lag=12):
    """
    Calcula la correlación cruzada con retraso (lag) entre variables climáticas y casos de dengue.
    VERSIÓN FINAL: Corrige el AttributeError final.
    """
    climate_vars = ['lluvia_mean', 'temperatura_mean']
    all_results = []

    if df['id_bar'].nunique() == 0:
        print("ADVERTENCIA: El DataFrame está vacío o no contiene la columna 'id_bar'.")
        return pd.DataFrame()

    for barrio in df['id_bar'].unique():
        barrio_data = df[df['id_bar'] == barrio].sort_values(['anio', 'semana']).copy()
        
        for var in climate_vars:
            for lag in range(max_lag + 1):
                dengue_series = barrio_data[dengue_col]
                climate_series_lagged = barrio_data[var].shift(lag)
                combined = pd.concat([dengue_series, climate_series_lagged], axis=1).dropna()
                
                if len(combined) < 3:
                    continue
                    
                corr, p_value = pearsonr(combined[dengue_col], combined[var])
                
                all_results.append({
                    'id_bar': barrio, 'variable': var, 'lag_semanas': lag,
                    'correlacion': corr, 'p_value': p_value
                })

    if not all_results:
        print("ADVERTENCIA: No se pudieron calcular correlaciones, posiblemente por datos insuficientes.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_results)
    
    results_df['abs_corr'] = results_df['correlacion'].abs()
    optimal_indices = results_df.loc[results_df.groupby(['id_bar', 'variable'])['abs_corr'].idxmax()].index
    
    # --- LÍNEA CORREGIDA ---
    # Se eliminó el ".index" extra al final de "optimal_indices".
    results_df['lag_optimo'] = results_df.index.isin(optimal_indices)
    
    results_df = results_df.drop(columns=['abs_corr'])
    
    return results_df

# --------------------------------------------
# Función para Identificar Correlación Climática con Retraso (Lag)
# --------------------------------------------

def spatial_correlation_analysis(df, dengue_col='dengue'):
    """
    Calcula la matriz de correlación de las series de tiempo de dengue entre todos los barrios.

    Args:
        df (pd.DataFrame): DataFrame con las columnas 'id_bar', 'anio', 'semana' y la de dengue.
        dengue_col (str): Nombre de la columna de casos de dengue.

    Returns:
        pd.DataFrame: Una matriz de correlación donde las filas y columnas son los 'id_bar'.
                      Un valor cercano a 1 indica una fuerte correlación positiva (patrones similares).
    """
    # 1. Crear una columna de fecha para usar como índice temporal
    df_temp = df.copy()
    df_temp['fecha'] = pd.to_datetime(df_temp['anio'].astype(str) + df_temp['semana'].astype(str).str.zfill(2) + '1', format='%Y%W%w')
    
    # 2. Pivotar la tabla
    # Queremos una estructura donde las columnas son los barrios y las filas son los casos de dengue a lo largo del tiempo.
    df_pivot = df_temp.pivot_table(
        index='fecha',
        columns='id_bar',
        values=dengue_col
    )
    
    # 3. Rellenar valores faltantes
    # Interpolar para rellenar huecos en las series de tiempo de cada barrio.
    df_pivot_filled = df_pivot.interpolate(method='time')
    
    # 4. Calcular la matriz de correlación de Pearson
    # Compara cada columna (barrio) con todas las demás.
    correlation_matrix = df_pivot_filled.corr(method='pearson')
    
    return correlation_matrix