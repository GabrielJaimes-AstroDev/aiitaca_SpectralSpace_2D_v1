import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import re
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
from tqdm import tqdm

# Set page configuration
st.set_page_config(
    page_title="Molecular Spectrum Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; margin-bottom: 1rem;}
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem; margin-top: 1.5rem;}
    .info-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

def sanitize_filename(filename):
    """Elimina caracteres inválidos de los nombres de archivo"""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def load_model(model_file):
    """Carga el modelo entrenado desde un archivo subido"""
    try:
        model = pickle.load(model_file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_molecule_formula(header):
    """
    Extract molecule formula from header string.
    Example: "molecules='C2H5OH,V=0|1'" returns "C2H5OH"
    """
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

def load_and_interpolate_spectrum(file_content, filename, reference_frequencies):
    """Carga un espectro desde contenido de archivo y lo interpola a las frecuencias de referencia"""
    try:
        # Convertir el contenido a líneas (manejar diferentes tipos de entrada)
        if isinstance(file_content, bytes):
            # Si es bytes, decodificar
            try:
                content = file_content.decode('utf-8')
            except UnicodeDecodeError:
                content = file_content.decode('latin-1')
        else:
            # Si ya es string
            content = file_content
            
        lines = content.splitlines()
        
        # Resto del código igual que en tu script original...
        first_line = lines[0].strip() if lines else ""
        second_line = lines[1].strip() if len(lines) > 1 else ""
        
        formula = "Unknown"
        param_dict = {}
        data_start_line = 0
        
        # Formato 1: con header de molécula y parámetros
        if first_line.startswith('//') and 'molecules=' in first_line:
            header = first_line[2:].strip()  # Remove the '//'
            formula = extract_molecule_formula(header)
            
            # Extraer parámetros del header
            for part in header.split():
                if '=' in part:
                    try:
                        key, value = part.split('=')
                        key = key.strip()
                        value = value.strip("'")
                        if key in ['molecules', 'sourcesize']:
                            continue
                        try:
                            param_dict[key] = float(value)
                        except ValueError:
                            param_dict[key] = value
                    except:
                        continue
            data_start_line = 1
        
        # Formato 2: con header de columnas
        elif first_line.startswith('!') or first_line.startswith('#'):
            # Intentar extraer información del header si está disponible
            if 'molecules=' in first_line:
                formula = extract_molecule_formula(first_line)
            data_start_line = 1
        
        # Formato 3: sin header, solo datos
        else:
            data_start_line = 0
            formula = filename.split('.')[0]  # Usar nombre del archivo como fórmula

        spectrum_data = []
        for line in lines[data_start_line:]:
            line = line.strip()
            # Saltar líneas de comentario o vacías
            if not line or line.startswith('!') or line.startswith('#') or line.startswith('//'):
                continue
                
            try:
                parts = line.split()
                if len(parts) >= 2:
                    # Intentar diferentes formatos de números
                    try:
                        freq = float(parts[0])
                        intensity = float(parts[1])
                    except ValueError:
                        # Intentar con notación científica que pueda tener D instead of E
                        freq_str = parts[0].replace('D', 'E').replace('d', 'E')
                        intensity_str = parts[1].replace('D', 'E').replace('d', 'E')
                        freq = float(freq_str)
                        intensity = float(intensity_str)
                    
                    if np.isfinite(freq) and np.isfinite(intensity):
                        spectrum_data.append([freq, intensity])
            except Exception as e:
                st.warning(f"Warning: Could not parse line '{line}' in {filename}: {e}")
                continue

        if not spectrum_data:
            raise ValueError("No valid data points found in spectrum file")

        spectrum_data = np.array(spectrum_data)

        # Ajustar frecuencia si está en GHz (convertir a Hz)
        if np.max(spectrum_data[:, 0]) < 1e11:  # Si las frecuencias son menores a 100 GHz, probablemente están en GHz
            spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convertir GHz to Hz
            st.info(f"Converted frequencies from GHz to Hz for {filename}")

        interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                                kind='linear', bounds_error=False, fill_value=0.0)
        interpolated = interpolator(reference_frequencies)

        # Extraer parámetros con valores por defecto si faltan
        params = [
            param_dict.get('logn', np.nan),
            param_dict.get('tex', np.nan),
            param_dict.get('velo', np.nan),
            param_dict.get('fwhm', np.nan)
        ]
        
        # Sanitize filename
        safe_filename = sanitize_filename(os.path.basename(filename))[:60]
        
        return spectrum_data, interpolated, formula, params, safe_filename

    except Exception as e:
        st.error(f"Error processing {filename}: {str(e)}")
        raise

def find_knn_neighbors(training_embeddings, new_embeddings, k=5):
    """Encuentra los k vecinos más cercanos usando KNN"""
    if len(training_embeddings) == 0 or len(new_embeddings) == 0:
        return []
    
    # Asegurar que k no sea mayor que el número de puntos de entrenamiento
    k = min(k, len(training_embeddings))
    
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(training_embeddings)
    
    all_neighbor_indices = []
    for new_embedding in new_embeddings:
        distances, indices = knn.kneighbors([new_embedding])
        # Verificar que los índices estén dentro del rango válido
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    
    return all_neighbor_indices

def create_safe_dataframe_for_plotting(model, results):
    """Create a safe DataFrame for plotting with proper data validation"""
    try:
        # Create training data
        train_data = {
            'umap_x': model['embedding'][:, 0].astype(float),
            'umap_y': model['embedding'][:, 1].astype(float),
            'formula': [str(f) for f in model['formulas']],
            'logn': model['y'][:, 0].astype(float),
            'tex': model['y'][:, 1].astype(float),
            'velo': model['y'][:, 2].astype(float),
            'fwhm': model['y'][:, 3].astype(float),
            'type': 'Training'
        }
        
        # Add filename to training data if available
        if 'filenames' in model and len(model['filenames']) == len(model['formulas']):
            train_data['filename'] = [str(f) for f in model['filenames']]
        
        train_df = pd.DataFrame(train_data)
        
        # Create new data if available
        if results and 'umap_embedding_new' in results and len(results['umap_embedding_new']) > 0:
            new_data = {
                'umap_x': results['umap_embedding_new'][:, 0].astype(float),
                'umap_y': results['umap_embedding_new'][:, 1].astype(float),
                'formula': [str(f) for f in results['formulas_new']],
                'logn': results['y_new'][:, 0].astype(float),
                'tex': results['y_new'][:, 1].astype(float),
                'velo': results['y_new'][:, 2].astype(float),
                'fwhm': results['y_new'][:, 3].astype(float),
                'filename': [str(f) for f in results['filenames_new']],
                'type': 'New'
            }
            
            new_df = pd.DataFrame(new_data)
            combined_df = pd.concat([train_df, new_df], ignore_index=True)
        else:
            combined_df = train_df
            
        # Ensure all columns are of proper type
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                combined_df[col] = combined_df[col].astype(str)
                
        return combined_df
        
    except Exception as e:
        st.error(f"Error creating DataFrame for plotting: {str(e)}")
        # Return empty DataFrame as fallback
        return pd.DataFrame()

def main():
    st.title("🧪 Molecular Spectrum Analyzer")
    st.markdown("""
    This interactive tool analyzes molecular spectra using a pre-trained machine learning model. 
    Upload your model and spectrum files to visualize the results.
    """)
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'spectra_files' not in st.session_state:
        st.session_state.spectra_files = []
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Input Parameters")
        
        # Model upload
        st.subheader("1. Upload Model")
        model_file = st.file_uploader("Upload trained model (PKL file)", type=['pkl'])
        
        if model_file is not None:
            if st.button("Load Model") or st.session_state.model is None:
                with st.spinner("Loading model..."):
                    st.session_state.model = load_model(model_file)
                    if st.session_state.model is not None:
                        st.success("Model loaded successfully!")
        
        # Spectra upload
        st.subheader("2. Upload Spectra")
        spectra_files = st.file_uploader("Upload spectrum files (TXT)", type=['txt'], accept_multiple_files=True)
        
        if spectra_files:
            st.session_state.spectra_files = spectra_files
        
        # Analysis parameters
        st.subheader("3. Analysis Parameters")
        knn_neighbors = st.slider("Number of KNN neighbors", min_value=1, max_value=20, value=5)
        
        if st.button("Analyze Spectra") and st.session_state.model is not None and st.session_state.spectra_files:
            with st.spinner("Analyzing spectra..."):
                try:
                    model = st.session_state.model
                    results = analyze_spectra(model, st.session_state.spectra_files, knn_neighbors)
                    st.session_state.results = results
                    st.success("Analysis completed!")
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    
    # Main content area
    if st.session_state.model is None:
        st.info("Please upload a model file to get started.")
        return
    
    model = st.session_state.model
    
    # Display model information
    with st.expander("Model Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", len(model['formulas']))
        with col2:
            st.metric("PCA Components", model.get('n_components', 'N/A'))
        with col3:
            st.metric("Variance Threshold", f"{model.get('variance_threshold', 0.99)*100:.1f}%")
    
    if st.session_state.results is None:
        st.info("Upload spectrum files and click 'Analyze Spectra' to see results.")
        return
    
    results = st.session_state.results
    
    # Display results
    st.header("Analysis Results")
    
    # UMAP Visualization
    st.subheader("UMAP Projection")
    
    # Create safe DataFrame for plotting
    combined_df = create_safe_dataframe_for_plotting(model, results)
    
    if combined_df.empty:
        st.error("Could not create visualization data. Please check your model and data.")
        return
    
    # Create interactive UMAP plot with error handling
    try:
        # Check if filename column exists for hover data
        hover_data = ['logn', 'tex', 'velo', 'fwhm']
        if 'filename' in combined_df.columns:
            hover_data.append('filename')
            
        fig = px.scatter(combined_df, x='umap_x', y='umap_y', color='formula', 
                         symbol='type', hover_data=hover_data,
                         title='UMAP Projection of Molecular Spectra')
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating UMAP plot: {str(e)}")
        # Fallback: show simple scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        for formula in combined_df['formula'].unique():
            mask = combined_df['formula'] == formula
            ax.scatter(combined_df.loc[mask, 'umap_x'], combined_df.loc[mask, 'umap_y'], 
                      label=formula, alpha=0.6)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('UMAP Projection of Molecular Spectra')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Parameter distribution plots
    st.subheader("Parameter Distributions")
    
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    # Create subplots for each parameter
    try:
        fig = make_subplots(rows=2, cols=2, subplot_titles=param_labels)
        
        for i, param in enumerate(param_names):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            # Add training data
            train_mask = combined_df['type'] == 'Training'
            fig.add_trace(
                go.Histogram(x=combined_df.loc[train_mask, param], name='Training', opacity=0.7, marker_color='blue'),
                row=row, col=col
            )
            
            # Add new data if available
            new_mask = combined_df['type'] == 'New'
            if new_mask.any():
                fig.add_trace(
                    go.Histogram(x=combined_df.loc[new_mask, param], name='New', opacity=0.7, marker_color='red'),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, showlegend=False, title_text="Parameter Distributions")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating parameter distribution plots: {str(e)}")
    
    # Individual spectrum analysis
    if 'umap_embedding_new' in results and len(results['umap_embedding_new']) > 0:
        st.subheader("Individual Spectrum Analysis")
        
        # Select a spectrum to analyze
        selected_idx = st.selectbox("Select a spectrum for detailed analysis", 
                                   range(len(results['filenames_new'])),
                                   format_func=lambda i: results['filenames_new'][i])
        
        if selected_idx is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Show spectrum plot
                st.markdown("**Spectrum Visualization**")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(model['reference_frequencies'], results['X_new'][selected_idx])
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Intensity')
                ax.set_title(f"Spectrum: {results['filenames_new'][selected_idx]}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Show parameters
                st.markdown("**Estimated Parameters**")
                param_data = {
                    'Parameter': param_labels,
                    'Value': [
                        results['y_new'][selected_idx, 0],
                        results['y_new'][selected_idx, 1],
                        results['y_new'][selected_idx, 2],
                        results['y_new'][selected_idx, 3]
                    ]
                }
                st.table(pd.DataFrame(param_data))
                
                # Show molecule formula
                st.markdown(f"**Molecule Formula**: {results['formulas_new'][selected_idx]}")
            
            # KNN Neighbors analysis
            st.markdown("**K-Nearest Neighbors Analysis**")
            
            if 'knn_neighbors' in results and selected_idx < len(results['knn_neighbors']):
                neighbor_indices = results['knn_neighbors'][selected_idx]
                
                if neighbor_indices:
                    # Create table of neighbors
                    neighbor_data = []
                    for idx in neighbor_indices:
                        neighbor_data.append({
                            'Formula': model['formulas'][idx],
                            'log(n)': f"{model['y'][idx, 0]:.2f}",
                            'T_ex (K)': f"{model['y'][idx, 1]:.2f}",
                            'Velocity': f"{model['y'][idx, 2]:.2f}",
                            'FWHM': f"{model['y'][idx, 3]:.2f}"
                        })
                    
                    st.table(pd.DataFrame(neighbor_data))
                else:
                    st.info("No neighbors found for this spectrum.")
            else:
                st.info("KNN analysis not available for this spectrum.")
    
    # Download results
    st.subheader("Download Results")
    
    if 'umap_embedding_new' in results and len(results['umap_embedding_new']) > 0 and st.button("Export Results to CSV"):
        # Create results dataframe
        results_df = pd.DataFrame({
            'filename': results['filenames_new'],
            'formula': results['formulas_new'],
            'umap_x': results['umap_embedding_new'][:, 0],
            'umap_y': results['umap_embedding_new'][:, 1],
            'logn': results['y_new'][:, 0],
            'tex': results['y_new'][:, 1],
            'velo': results['y_new'][:, 2],
            'fwhm': results['y_new'][:, 3]
        })
        
        # Convert to CSV
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="spectrum_analysis_results.csv",
            mime="text/csv"
        )

def analyze_spectra(model, spectra_files, knn_neighbors=5):
    """Analyze uploaded spectra using the trained model"""
    results = {
        'X_new': [],
        'y_new': [],
        'formulas_new': [],
        'filenames_new': [],
        'pca_components_new': [],
        'umap_embedding_new': [],
        'knn_neighbors': []
    }
    
    # Get model components
    scaler = model['scaler']
    pca = model['pca']
    umap_model = model['umap']
    ref_freqs = model['reference_frequencies']
    
    # Process each spectrum
    successful_files = 0
    for spectrum_file in spectra_files:
        try:
            # Read file content once
            file_content = spectrum_file.getvalue()
            spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                file_content, spectrum_file.name, ref_freqs
            )
            
            # Transform the spectrum
            X_scaled = scaler.transform([interpolated])
            X_pca = pca.transform(X_scaled)
            X_umap = umap_model.transform(X_pca)
            
            results['X_new'].append(interpolated)
            results['formulas_new'].append(formula)
            results['y_new'].append(params)
            results['filenames_new'].append(filename)
            results['umap_embedding_new'].append(X_umap[0])
            results['pca_components_new'].append(X_pca[0])
            
            successful_files += 1
            
        except Exception as e:
            st.warning(f"Error processing {spectrum_file.name}: {str(e)}")
            continue
    
    st.info(f"Successfully processed {successful_files} out of {len(spectra_files)} spectrum files.")
    
    # Convert to arrays
    if results['umap_embedding_new']:
        results['X_new'] = np.array(results['X_new'])
        results['y_new'] = np.array(results['y_new'])
        results['formulas_new'] = np.array(results['formulas_new'])
        results['umap_embedding_new'] = np.array(results['umap_embedding_new'])
        results['pca_components_new'] = np.array(results['pca_components_new'])
        
        # Find KNN neighbors
        results['knn_neighbors'] = find_knn_neighbors(
            model['embedding'], results['umap_embedding_new'], k=knn_neighbors
        )
    else:
        st.error("No valid spectra could be processed. Please check your spectrum files.")
    
    return results

if __name__ == "__main__":
    main()


