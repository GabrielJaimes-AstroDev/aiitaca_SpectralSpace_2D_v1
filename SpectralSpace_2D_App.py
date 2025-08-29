import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import re
from io import StringIO
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import traceback

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Parámetros de Espectros",
    page_icon="📊",
    layout="wide"
)

# Título de la aplicación
st.title("📊 Predictor de Parámetros de Espectros Moleculares")
st.markdown("""
Esta aplicación utiliza un modelo preentrenado de PCA + UMAP para predecir los parámetros físicos 
(logn, tex, velo, fwhm) de espectros moleculares a partir de archivos de texto.
""")

# Función para extraer fórmula molecular (modificada para nuevos formatos)
def extract_molecule_info(filename):
    """
    Extrae información de la molécula del nombre del archivo.
    Ejemplo: "CH3OH_spectrum.txt" → "CH3OH"
    """
    # Eliminar extensiones y números al final
    name = os.path.splitext(filename)[0]
    # Buscar patrones comunes de fórmulas moleculares
    pattern = r'([A-Z][a-z]?\d*[A-Z][a-z]?\d*)'
    match = re.search(pattern, name)
    if match:
        return match.group(1)
    return "Unknown"

# Función para procesar espectros de predicción (nuevo formato)
def process_prediction_spectrum(file_content, filename, reference_frequencies):
    """
    Procesa un espectro en el formato de predicción.
    
    Args:
        file_content (str): Contenido del archivo
        filename (str): Nombre del archivo
        reference_frequencies (array): Frecuencias de referencia del modelo
    
    Returns:
        tuple: (espectro interpolado, fórmula, nombre de archivo)
    """
    try:
        lines = file_content.split('\n')
        spectrum_data = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('!') and not line.startswith('//'):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        # Convertir a float y manejar notación científica
                        freq = float(parts[0])
                        intensity = float(parts[1])
                        if np.isfinite(freq) and np.isfinite(intensity):
                            spectrum_data.append([freq, intensity])
                    except ValueError:
                        continue
        
        if not spectrum_data:
            raise ValueError("No valid data points found")
            
        spectrum_data = np.array(spectrum_data)
        
        # Interpolar al espacio de frecuencia de referencia
        interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1], 
                               kind='linear', bounds_error=False, fill_value=0.0)
        interpolated = interpolator(reference_frequencies)
        
        if not np.all(np.isfinite(interpolated)):
            raise ValueError("Invalid values after interpolation")
            
        # Extraer información de la molécula del nombre del archivo
        formula = extract_molecule_info(filename)
            
        return interpolated, formula, filename
        
    except Exception as e:
        raise ValueError(f"Error processing file {filename}: {str(e)}")

# Función para cargar el modelo con manejo robusto de errores
@st.cache_resource
def load_model(model_file):
    """Carga el modelo preentrenado desde un archivo .pkl con manejo robusto de errores"""
    try:
        # Leer el contenido del archivo
        model_content = model_file.read()
        
        # Intentar cargar con diferentes protocolos de pickle
        try:
            model = pickle.loads(model_content)
        except:
            # Intentar con encoding latin1 para compatibilidad
            model = pickle.loads(model_content, encoding='latin1')
        
        # Verificar que el modelo tiene la estructura esperada
        required_keys = ['scaler', 'pca', 'umap', 'reference_frequencies']
        for key in required_keys:
            if key not in model:
                st.error(f"El modelo no contiene la clave requerida: {key}")
                return None
                
        st.success("✅ Modelo cargado exitosamente")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None

# Función para hacer predicciones
def predict_spectra(model, spectra_data):
    """
    Realiza predicciones para nuevos espectros
    
    Args:
        model: Modelo cargado
        spectra_data: Lista de tuplas (spectrum, formula, filename)
    
    Returns:
        DataFrame con resultados y embeddings
    """
    try:
        # Extraer los espectros
        X_new = np.array([data[0] for data in spectra_data])
        formulas = [data[1] for data in spectra_data]
        filenames = [data[2] for data in spectra_data]
        
        # Preprocesamiento
        X_new_scaled = model['scaler'].transform(X_new)
        pca_components_new = model['pca'].transform(X_new_scaled)
        
        # Transformación UMAP
        try:
            # Intentar transformación UMAP normal
            umap_embedding_new = model['umap'].transform(pca_components_new)
        except:
            # Si falla, usar método alternativo
            st.warning("Usando método alternativo para transformación UMAP")
            if 'X_pca_train' in model:
                # Concatenar con datos de entrenamiento para consistencia
                combined_data = np.vstack([model['X_pca_train'], pca_components_new])
                combined_embedding = model['umap'].transform(combined_data)
                umap_embedding_new = combined_embedding[len(model['X_pca_train']):]
            else:
                # Método simple si no hay datos de entrenamiento
                umap_embedding_new = pca_components_new[:, :2]  # Usar primeras 2 componentes PCA
        
        # Predecir parámetros basados en la posición en el espacio de características
        # Esta es una aproximación simplificada - en producción usarías un modelo regresor
        
        # Obtener rangos de parámetros del modelo de entrenamiento si están disponibles
        if 'y' in model and model['y'] is not None and len(model['y']) > 0:
            y_train = model['y']
            logn_range = [np.min(y_train[:, 0]), np.max(y_train[:, 0])] if len(y_train) > 0 else [12.0, 16.0]
            tex_range = [np.min(y_train[:, 1]), np.max(y_train[:, 1])] if len(y_train) > 0 else [10.0, 300.0]
            velo_range = [np.min(y_train[:, 2]), np.max(y_train[:, 2])] if len(y_train) > 0 else [0.0, 10.0]
            fwhm_range = [np.min(y_train[:, 3]), np.max(y_train[:, 3])] if len(y_train) > 0 else [1.0, 10.0]
        else:
            # Valores por defecto si no hay datos de entrenamiento
            logn_range = [12.0, 16.0]
            tex_range = [10.0, 300.0]
            velo_range = [0.0, 10.0]
            fwhm_range = [1.0, 10.0]
        
        # Normalizar las coordenadas UMAP/PCA al rango [0, 1]
        if len(umap_embedding_new) > 0:
            umap_x_norm = (umap_embedding_new[:, 0] - np.min(umap_embedding_new[:, 0])) / (np.max(umap_embedding_new[:, 0]) - np.min(umap_embedding_new[:, 0]) + 1e-10)
            umap_y_norm = (umap_embedding_new[:, 1] - np.min(umap_embedding_new[:, 1])) / (np.max(umap_embedding_new[:, 1]) - np.min(umap_embedding_new[:, 1]) + 1e-10)
        else:
            umap_x_norm = np.array([0.5])
            umap_y_norm = np.array([0.5])
        
        # Crear DataFrame con resultados
        results = []
        for i, (formula, filename) in enumerate(zip(formulas, filenames)):
            # Estimación basada en posición en el espacio de características
            # Esta es una aproximación simplificada
            logn_est = logn_range[0] + umap_x_norm[i] * (logn_range[1] - logn_range[0])
            tex_est = tex_range[0] + umap_y_norm[i] * (tex_range[1] - tex_range[0])
            velo_est = velo_range[0] + umap_x_norm[i] * (velo_range[1] - velo_range[0])
            fwhm_est = fwhm_range[0] + umap_y_norm[i] * (fwhm_range[1] - fwhm_range[0])
            
            results.append({
                'Filename': filename,
                'Formula': formula,
                'UMAP_X': umap_embedding_new[i, 0] if i < len(umap_embedding_new) else 0,
                'UMAP_Y': umap_embedding_new[i, 1] if i < len(umap_embedding_new) else 0,
                'logn_estimated': logn_est,
                'tex_estimated': tex_est,
                'velo_estimated': velo_est,
                'fwhm_estimated': fwhm_est,
                'PCA_Components': pca_components_new[i].tolist() if i < len(pca_components_new) else []
            })
        
        return pd.DataFrame(results), umap_embedding_new, pca_components_new
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.error("Traceback:")
        st.code(traceback.format_exc())
        return None, None, None

# Función para visualizar resultados
def create_visualizations(model, df_results, umap_embedding_new, pca_components_new):
    """Crea visualizaciones interactivas con Plotly"""
    
    # 1. UMAP con datos de entrenamiento y nuevas predicciones
    fig_umap = go.Figure()
    
    # Datos de entrenamiento (si están disponibles)
    if 'embedding' in model and model['embedding'] is not None and len(model['embedding']) > 0:
        fig_umap.add_trace(go.Scatter(
            x=model['embedding'][:, 0],
            y=model['embedding'][:, 1],
            mode='markers',
            name='Training Data',
            marker=dict(color='lightblue', size=6, opacity=0.5),
            hovertemplate='<b>Training</b><br>UMAP1: %{x}<br>UMAP2: %{y}<extra></extra>'
        ))
    
    # Nuevas predicciones
    fig_umap.add_trace(go.Scatter(
        x=df_results['UMAP_X'],
        y=df_results['UMAP_Y'],
        mode='markers+text',
        name='New Predictions',
        marker=dict(color='red', size=12, symbol='star'),
        text=df_results['Formula'],
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>Filename: %{customdata[0]}<br>UMAP1: %{x}<br>UMAP2: %{y}<extra></extra>',
        customdata=df_results[['Filename']]
    ))
    
    fig_umap.update_layout(
        title='UMAP Projection: Training Data vs New Predictions',
        xaxis_title='UMAP 1',
        yaxis_title='UMAP 2',
        hovermode='closest'
    )
    
    # 2. Gráfico de parámetros predichos
    param_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)')
    )
    
    params = ['logn_estimated', 'tex_estimated', 'velo_estimated', 'fwhm_estimated']
    titles = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    for i, param in enumerate(params):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        param_fig.add_trace(go.Bar(
            x=df_results['Formula'],
            y=df_results[param],
            name=titles[i],
            text=df_results[param].round(2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>%{y:.2f}<extra></extra>'
        ), row=row, col=col)
    
    param_fig.update_layout(
        title_text='Parámetros Predichos por Molécula',
        showlegend=False,
        height=600
    )
    
    # 3. Espectros cargados (primeros 5)
    spectrum_fig = go.Figure()
    if 'reference_frequencies' in model and model['reference_frequencies'] is not None:
        for i, (idx, row) in enumerate(df_results.head(5).iterrows()):
            if i < len(pca_components_new):
                try:
                    # Reconstruir el espectro escalado inverso
                    reconstructed = model['scaler'].inverse_transform([pca_components_new[i]])[0]
                    spectrum_fig.add_trace(go.Scatter(
                        x=model['reference_frequencies'],
                        y=reconstructed,
                        mode='lines',
                        name=f"{row['Formula']} ({row['Filename']})",
                        hovertemplate='<b>%{fullData.name}</b><br>Freq: %{x}<br>Intensity: %{y}<extra></extra>'
                    ))
                except:
                    continue
    
    spectrum_fig.update_layout(
        title='Espectros Interpolados (Primeros 5)',
        xaxis_title='Frecuencia',
        yaxis_title='Intensidad'
    )
    
    return fig_umap, param_fig, spectrum_fig

# Interfaz principal de la aplicación
def main():
    # Sidebar para carga de archivos
    with st.sidebar:
        st.header("📁 Cargar Modelo y Datos")
        
        # Cargar modelo
        model_file = st.file_uploader("Cargar modelo preentrenado (.pkl)", type="pkl")
        
        # Cargar espectros
        spectra_files = st.file_uploader("Cargar espectros para predecir (.txt)", 
                                        type="txt", accept_multiple_files=True)
        
        # Botón para procesar
        process_btn = st.button("🚀 Procesar Espectros", type="primary")
    
    # Panel principal
    if model_file is not None:
        # Cargar modelo
        with st.spinner("Cargando modelo..."):
            model = load_model(model_file)
        
        if model is not None:
            # Mostrar información del modelo
            with st.expander("📋 Información del Modelo"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Parámetros:**")
                    st.write(f"- Componentes PCA: {model.get('n_components', 'N/A')}")
                    st.write(f"- Umbral de varianza: {model.get('variance_threshold', 'N/A')}")
                    st.write(f"- Longitud objetivo: {model.get('target_length', 'N/A')}")
                
                with col2:
                    st.write("**Rangos de parámetros:**")
                    if 'y' in model and model['y'] is not None and len(model['y']) > 0:
                        y = model['y']
                        st.write(f"- logn: {y[:, 0].min():.2f} to {y[:, 0].max():.2f}")
                        st.write(f"- tex: {y[:, 1].min():.2f} to {y[:, 1].max():.2f}")
                        st.write(f"- velo: {y[:, 2].min():.2f} to {y[:, 2].max():.2f}")
                        st.write(f"- fwhm: {y[:, 3].min():.2f} to {y[:, 3].max():.2f}")
                    else:
                        st.write("No hay datos de entrenamiento disponibles")
            
            # Procesar espectros si se han cargado
            if spectra_files and process_btn:
                with st.spinner("Procesando espectros..."):
                    spectra_data = []
                    for spectra_file in spectra_files:
                        try:
                            # Leer contenido del archivo
                            content = spectra_file.getvalue().decode("utf-8")
                            # Procesar espectro
                            spectrum, formula, filename = process_prediction_spectrum(
                                content, spectra_file.name, model['reference_frequencies']
                            )
                            spectra_data.append((spectrum, formula, filename))
                        except Exception as e:
                            st.error(f"Error procesando {spectra_file.name}: {str(e)}")
                    
                    if spectra_data:
                        st.success(f"✅ {len(spectra_data)} espectros procesados correctamente")
                        
                        # Realizar predicciones
                        df_results, umap_embedding, pca_components = predict_spectra(model, spectra_data)
                        
                        if df_results is not None:
                            # Mostrar advertencia sobre las predicciones
                            st.warning("""
                            ⚠️ **Nota importante:** Las predicciones mostradas son estimaciones basadas en la posición en el espacio de características.
                            Para predicciones precisas, se necesita un modelo regresor adicional entrenado específicamente en los parámetros físicos.
                            """)
                            
                            # Mostrar resultados en tabla
                            st.subheader("📊 Resultados de Predicción")
                            
                            # Crear tabla resumen
                            summary_df = df_results[['Filename', 'Formula', 'logn_estimated', 
                                                    'tex_estimated', 'velo_estimated', 'fwhm_estimated']].copy()
                            summary_df.columns = ['Archivo', 'Fórmula', 'log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
                            
                            # Formatear números
                            for col in ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']:
                                summary_df[col] = summary_df[col].round(3)
                            
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Botón para descargar resultados
                            csv = summary_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Descargar resultados CSV",
                                data=csv,
                                file_name="predicciones_espectros.csv",
                                mime="text/csv"
                            )
                            
                            # Visualizaciones
                            st.subheader("📈 Visualizaciones")
                            
                            fig_umap, param_fig, spectrum_fig = create_visualizations(
                                model, df_results, umap_embedding, pca_components
                            )
                            
                            st.plotly_chart(fig_umap, use_container_width=True)
                            st.plotly_chart(param_fig, use_container_width=True)
                            st.plotly_chart(spectrum_fig, use_container_width=True)
                            
                        else:
                            st.error("Error al realizar las predicciones")
                    else:
                        st.warning("No se pudieron procesar los espectros")
        else:
            st.error("Error al cargar el modelo. Verifica que el archivo .pkl sea válido.")
    else:
        # Instrucciones cuando no hay modelo cargado
        st.info("👈 Por favor, carga un modelo preentrenado en la barra lateral para comenzar.")
        
        # Mostrar ejemplo de formato de archivo
        with st.expander("📝 Formato esperado de los archivos de espectros"):
            st.code("""
            !xValues(GHz)	yValues(K)
            84.0797306920	0.000000e+00
            84.0802239790	0.000000e+00
            84.0807172650	0.000000e+00
            84.0812105520	0.000000e+00
            84.0817038380	0.000000e+00
            ... (más puntos)
            """)
            
            st.write("""
            **Recomendaciones:**
            - Los archivos deben ser de texto plano (.txt)
            - La primera línea puede ser un encabezado que comience con ! o //
            - Las líneas de datos deben tener dos columnas: frecuencia e intensidad
            - Las frecuencias pueden estar en GHz, MHz o Hz, pero deben ser consistentes
            """)

if __name__ == "__main__":
    main()
