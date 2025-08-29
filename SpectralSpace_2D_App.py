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

# Función para cargar el modelo
@st.cache_resource
def load_model(model_file):
    """Carga el modelo preentrenado desde un archivo .pkl"""
    try:
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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
        
        # Transformación UMAP (método seguro)
        if 'X_pca_train' in model:
            # Concatenar con datos de entrenamiento para consistencia
            combined_data = np.vstack([model['X_pca_train'], pca_components_new])
            combined_embedding = model['umap'].transform(combined_data)
            umap_embedding_new = combined_embedding[len(model['X_pca_train']):]
        else:
            # Método alternativo
            umap_embedding_new = model['umap'].transform(pca_components_new)
        
        # Predecir parámetros usando el embedding (esto es un placeholder)
        # En un caso real, necesitarías un modelo regresor entrenado en el espacio UMAP
        st.warning("""
        ⚠️ Nota: Las predicciones mostradas son estimaciones basadas en la posición en el espacio UMAP.
        Para predicciones precisas, se necesita un modelo regresor adicional entrenado en los parámetros físicos.
        """)
        
        # Crear DataFrame con resultados
        results = []
        for i, (formula, filename) in enumerate(zip(formulas, filenames)):
            # Estimación basada en posición en el espacio UMAP (esto es simplificado)
            # En una implementación real, usarías un modelo de regresión
            results.append({
                'Filename': filename,
                'Formula': formula,
                'UMAP_X': umap_embedding_new[i, 0],
                'UMAP_Y': umap_embedding_new[i, 1],
                'logn_estimated': 13.0 + umap_embedding_new[i, 0] * 0.5,  # Placeholder
                'tex_estimated': 50.0 + umap_embedding_new[i, 1] * 20.0,   # Placeholder
                'velo_estimated': umap_embedding_new[i, 0] * 2.0,          # Placeholder
                'fwhm_estimated': 5.0 + umap_embedding_new[i, 1] * 1.0,    # Placeholder
                'PCA_Components': pca_components_new[i].tolist()
            })
        
        return pd.DataFrame(results), umap_embedding_new, pca_components_new
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# Función para visualizar resultados
def create_visualizations(model, df_results, umap_embedding_new, pca_components_new):
    """Crea visualizaciones interactivas con Plotly"""
    
    # 1. UMAP con datos de entrenamiento y nuevas predicciones
    fig_umap = go.Figure()
    
    # Datos de entrenamiento
    if 'embedding' in model:
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
    for i, (idx, row) in enumerate(df_results.head(5).iterrows()):
        spectrum_fig.add_trace(go.Scatter(
            x=model['reference_frequencies'],
            y=model['scaler'].inverse_transform([pca_components_new[i]])[0],
            mode='lines',
            name=f"{row['Formula']} ({row['Filename']})",
            hovertemplate='<b>%{fullData.name}</b><br>Freq: %{x}<br>Intensity: %{y}<extra></extra>'
        ))
    
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
            st.success(f"✅ Modelo cargado: {len(model.get('X', []))} espectros de entrenamiento")
            
            # Mostrar información del modelo
            with st.expander("📋 Información del Modelo"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Parámetros:**")
                    st.write(f"- Muestras de entrenamiento: {model.get('sample_size', 'N/A')}")
                    st.write(f"- Componentes PCA: {model.get('n_components', 'N/A')}")
                    st.write(f"- Umbral de varianza: {model.get('variance_threshold', 'N/A')}")
                
                with col2:
                    st.write("**Rangos de parámetros:**")
                    if 'y' in model:
                        y = model['y']
                        st.write(f"- logn: {y[:, 0].min():.2f} to {y[:, 0].max():.2f}")
                        st.write(f"- tex: {y[:, 1].min():.2f} to {y[:, 1].max():.2f}")
                        st.write(f"- velo: {y[:, 2].min():.2f} to {y[:, 2].max():.2f}")
                        st.write(f"- fwhm: {y[:, 3].min():.2f} to {y[:, 3].max():.2f}")
            
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
            st.error("Error al cargar el modelo")
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
