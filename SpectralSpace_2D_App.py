import streamlit as st
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from tqdm import tqdm
import re
from sklearn.neighbors import NearestNeighbors
import tempfile
import base64
from io import BytesIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import traceback
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Molecular Spectrum Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
    .debug-info {
        background-color: #fff3cd;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def sanitize_filename(filename):
    """Elimina caracteres inválidos de los nombres de archivo"""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def safe_pickle_load(file_obj):
    """
    Carga un archivo pickle de forma segura, manejando problemas de compatibilidad
    con objetos Numba y otros problemas de serialización.
    """
    try:
        # Primer intento: carga normal
        return pickle.load(file_obj)
    except (KeyError, AttributeError, TypeError) as e:
        st.warning(f"Primer intento falló: {e}. Intentando con manejo de errores...")
        
        # Segundo intento: con manejo de errores específico
        try:
            # Reiniciar el cursor del archivo
            file_obj.seek(0)
            
            # Usar pickle con encoding específico y manejo de errores
            return pickle.load(file_obj, encoding='latin1', errors='ignore')
        except Exception as e2:
            st.error(f"Segundo intento falló: {e2}")
            
            # Tercer intento: cargar manualmente evitando objetos problemáticos
            try:
                file_obj.seek(0)
                data = {}
                
                # Leer el contenido como bytes y reconstruir manualmente
                content = file_obj.read()
                
                # Buscar y extraer componentes clave del modelo
                # Esta es una aproximación genérica - deberías adaptarla a tu estructura de modelo específica
                if b'reference_frequencies' in content:
                    # Intentar extraer datos numéricos
                    st.info("Intentando reconstrucción manual del modelo...")
                    
                    # Aquí necesitarías conocer la estructura exacta de tu modelo
                    # Esto es solo un ejemplo genérico
                    model_structure = {
                        'reference_frequencies': None,
                        'scaler': None,
                        'pca': None,
                        'umap': None,
                        'embedding': None,
                        'formulas': None,
                        'y': None,
                        'cumulative_variance': None,
                        'n_components': None,
                        'variance_threshold': None,
                        'sample_size': None
                    }
                    
                    st.warning("Reconstrucción manual necesaria. Por favor implementa según tu estructura de modelo.")
                    return model_structure
                    
            except Exception as e3:
                st.error(f"Todos los intentos fallaron: {e3}")
                raise

def load_model(uploaded_model):
    """Carga el modelo entrenado desde un archivo subido"""
    try:
        # Guardar el archivo temporalmente para múltiples intentos de lectura
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Intentar diferentes enfoques para cargar el pickle
            with open(tmp_path, 'rb') as f:
                # Primer intento: carga normal
                model = pickle.load(f)
                
        except (KeyError, AttributeError, TypeError) as e:
            st.warning(f"Error inicial al cargar modelo: {e}")
            st.info("Intentando enfoques alternativos...")
            
            # Segundo intento: con encoding diferente
            with open(tmp_path, 'rb') as f:
                try:
                    model = pickle.load(f, encoding='latin1')
                except:
                    # Tercer intento: ignorar errores
                    f.seek(0)
                    model = pickle.load(f, errors='ignore')
        
        # Limpiar archivo temporal
        os.unlink(tmp_path)
        
        # Verificar estructura básica del modelo
        required_keys = ['reference_frequencies', 'scaler', 'pca', 'embedding', 'formulas', 'y']
        for key in required_keys:
            if key not in model:
                st.error(f"Modelo no contiene la clave requerida: {key}")
                return None
        
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Información adicional para debugging
        st.subheader("🔍 Debug Information")
        st.write("El error KeyError: 90 sugiere un problema de compatibilidad con objetos Numba en el pickle.")
        st.write("**Posibles soluciones:**")
        st.write("1. Entrenar el modelo en el mismo entorno donde se ejecuta la app")
        st.write("2. Usar una versión compatible de Numba")
        st.write("3. Excluir objetos Numba del pickle al guardar el modelo")
        
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

def inspect_spectrum_file(file_content, filename):
    """Inspecciona el contenido del archivo para debugging"""
    st.subheader(f"🔍 File Inspection: {filename}")
    
    content = file_content.getvalue().decode('utf-8')
    lines = content.split('\n')
    
    st.write(f"**Total lines:** {len(lines)}")
    st.write(f"**First 5 lines:**")
    for i, line in enumerate(lines[:5]):
        st.write(f"{i}: {repr(line)}")
    
    st.write(f"**Last 5 lines:**")
    for i, line in enumerate(lines[-5:]):
        st.write(f"{len(lines)-5+i}: {repr(line)}")
    
    # Check for non-numeric values
    numeric_lines = 0
    problematic_lines = []
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(('!', '#', '//')):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    float(parts[0].replace('D', 'E').replace('d', 'E'))
                    float(parts[1].replace('D', 'E').replace('d', 'E'))
                    numeric_lines += 1
                except Exception as e:
                    problematic_lines.append((i, line, str(e)))
    
    st.write(f"**Numeric data lines:** {numeric_lines}")
    
    if problematic_lines:
        st.write(f"**Problematic lines ({len(problematic_lines)}):**")
        for i, line, error in problematic_lines[:10]:  # Show first 10 problematic lines
            st.write(f"Line {i}: {repr(line)} -> Error: {error}")

def load_and_interpolate_spectrum(file_content, reference_frequencies, filename):
    """Carga un espectro desde contenido de archivo y lo interpola"""
    try:
        # Decode file content
        content = file_content.getvalue().decode('utf-8')
        lines = content.split('\n')
        
        st.markdown(f'<div class="debug-info">🔍 DEBUG: Processing {filename}</div>', unsafe_allow_html=True)
        st.write(f"🔍 DEBUG: First few lines: {lines[:3]}")
        
        # Determinar el formato del archivo
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
        problematic_lines = []
        for line_num, line in enumerate(lines[data_start_line:], data_start_line):
            line = line.strip()
            # Saltar líneas de comentario o vacías
            if not line or line.startswith('!') or line.startswith('#'):
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
                    else:
                        problematic_lines.append((line_num, line, "Non-finite values"))
                else:
                    problematic_lines.append((line_num, line, "Not enough columns"))
            except Exception as e:
                problematic_lines.append((line_num, line, f"Parse error: {e}"))
                continue

        if problematic_lines:
            st.warning(f"Found {len(problematic_lines)} problematic lines in {filename}")
            for line_num, line, error in problematic_lines[:5]:  # Show first 5 problematic lines
                st.write(f"Line {line_num}: {repr(line)} -> {error}")

        if not spectrum_data:
            raise ValueError("No valid data points found in spectrum file")

        spectrum_data = np.array(spectrum_data)
        st.write(f"🔍 DEBUG: Spectrum data shape: {spectrum_data.shape}")

        # Ajustar frecuencia si está en GHz (convertir a Hz)
        max_freq = np.max(spectrum_data[:, 0])
        st.write(f"🔍 DEBUG: Max frequency before conversion: {max_freq}")
        
        if max_freq < 1e11:  # Si las frecuencias son menores a 100 GHz, probablemente están en GHz
            spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convertir GHz to Hz
            st.write(f"🔍 DEBUG: Max frequency after conversion: {np.max(spectrum_data[:, 0])}")
            st.info(f"Converted frequencies from GHz to Hz for {filename}")

        # Verificar que las frecuencias estén dentro del rango de referencia
        ref_min = np.min(reference_frequencies)
        ref_max = np.max(reference_frequencies)
        spec_min = np.min(spectrum_data[:, 0])
        spec_max = np.max(spectrum_data[:, 0])
        
        st.write(f"🔍 DEBUG: Reference frequencies range: {ref_min} to {ref_max}")
        st.write(f"🔍 DEBUG: Spectrum frequencies range: {spec_min} to {spec_max}")
        
        # Check if spectrum frequencies overlap with reference frequencies
        if spec_max < ref_min or spec_min > ref_max:
            st.error(f"❌ ERROR: Spectrum frequencies ({spec_min}-{spec_max}) do not overlap with reference frequencies ({ref_min}-{ref_max})")
            return None, None, None, None, None

        interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                                kind='linear', bounds_error=False, fill_value=0.0)
        interpolated = interpolator(reference_frequencies)
        
        st.write(f"🔍 DEBUG: Interpolation completed successfully")
        st.write(f"🔍 DEBUG: Interpolated shape: {interpolated.shape}")

        # Extraer parámetros con valores por defecto si faltan
        params = [
            param_dict.get('logn', np.nan),
            param_dict.get('tex', np.nan),
            param_dict.get('velo', np.nan),
            param_dict.get('fwhm', np.nan)
        ]

        if not np.all(np.isfinite(interpolated)):
            st.error(f"❌ ERROR: Interpolated spectrum for {filename} contains NaN or inf values.")
            return None, None, None, None, None

        return spectrum_data, interpolated, formula, params, filename
    
    except Exception as e:
        st.error(f"❌ ERROR processing {filename}: {str(e)}")
        st.error(f"❌ ERROR type: {type(e).__name__}")
        st.error(f"❌ ERROR traceback: {traceback.format_exc()}")
        return None, None, None, None, None

def plot_example_spectrum(file_content, reference_frequencies, filename):
    """Grafica el espectro original vs interpolado"""
    spectrum_data, interpolated, formula, params, _ = load_and_interpolate_spectrum(
        file_content, reference_frequencies, filename
    )
    
    if spectrum_data is None:
        return None
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spectrum_data[:, 0], spectrum_data[:, 1], label='Original', alpha=0.7)
    ax.plot(reference_frequencies, interpolated, label='Interpolated', lw=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Example Spectrum: {filename}\nFormula: {formula}')
    ax.legend()
    ax.grid(True, alpha=0.4)
    
    return fig

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

def safe_umap_transform(umap_model, training_data, new_data):
    """
    Transforma nuevos datos de forma segura con UMAP,
    manejando posibles errores de compatibilidad.
    """
    try:
        # Intentar transformación directa primero
        return umap_model.transform(new_data)
    except Exception as e:
        st.warning(f"UMAP direct transform failed: {e}. Using alternative approach.")
        
        try:
            # Alternativa: crear nuevo UMAP con mismos parámetros
            alt_umap = UMAP(
                n_components=umap_model.n_components,
                n_neighbors=umap_model.n_neighbors,
                min_dist=umap_model.min_dist,
                metric=umap_model.metric,
                random_state=42,
                low_memory=True
            )
            
            # Ajustar con datos de entrenamiento y transformar
            alt_umap.fit(training_data)
            return alt_umap.transform(new_data)
            
        except Exception as alt_e:
            st.error(f"Alternative UMAP also failed: {alt_e}")
            # Último recurso: usar PCA directamente
            st.info("Using PCA components as fallback for UMAP")
            return new_data[:, :2]  # Tomar solo las primeras 2 componentes

def process_spectra(model, uploaded_files, knn_neighbors=5):
    """Procesa los espectros cargados y genera resultados"""
    results = {}
    
    # Extraer componentes del modelo
    scaler = model['scaler']
    pca = model['pca']
    umap_model = model['umap']
    ref_freqs = model['reference_frequencies']
    cumulative_variance = model['cumulative_variance']
    n_components = model['n_components']
    
    st.write(f"🔍 DEBUG: Reference frequencies shape: {ref_freqs.shape}")
    st.write(f"🔍 DEBUG: Reference frequencies range: {ref_freqs[0]} to {ref_freqs[-1]}")
    
    # Procesar todos los nuevos espectros
    new_spectra_data = []
    new_formulas = []
    new_params = []
    new_filenames = []
    new_embeddings = []
    new_pca_components = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        try:
            spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                uploaded_file, ref_freqs, uploaded_file.name
            )
            
            if interpolated is not None:
                # Transformar el espectro
                X_scaled = scaler.transform([interpolated])
                X_pca = pca.transform(X_scaled)
                
                # Transformación UMAP robusta
                X_umap = safe_umap_transform(
                    umap_model,
                    model.get('X_pca_train', model.get('X_pca', np.array([]))),
                    X_pca
                )
                
                new_spectra_data.append(interpolated)
                new_formulas.append(formula)
                new_params.append(params)
                new_filenames.append(filename)
                new_embeddings.append(X_umap[0])
                new_pca_components.append(X_pca[0])
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            st.error(f"Error type: {type(e).__name__}")
            st.error(f"Traceback: {traceback.format_exc()}")
            continue

    if not new_embeddings:
        st.error("No valid spectra found for prediction")
        return None
        
    new_embeddings = np.array(new_embeddings)
    new_params = np.array(new_params)
    new_formulas = np.array(new_formulas)
    new_pca_components = np.array(new_pca_components)
    
    # Encontrar vecinos KNN
    knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=knn_neighbors)
    
    # Compilar resultados
    results = {
        'new_spectra_data': new_spectra_data,
        'new_embeddings': new_embeddings,
        'new_params': new_params,
        'new_formulas': new_formulas,
        'new_filenames': new_filenames,
        'new_pca_components': new_pca_components,
        'knn_indices': knn_indices,
        'model': model
    }
    
    progress_bar.empty()
    status_text.empty()
    
    return results

# ... (el resto de las funciones create_visualizations y main se mantienen igual)

def main():
    """Main function for the Streamlit app"""
    st.markdown('<h1 class="main-header">🧪 Molecular Spectrum Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # File uploaders
        st.subheader("Upload Model")
        uploaded_model = st.file_uploader("Choose a trained model (.pkl)", type="pkl")
        
        st.subheader("Upload Spectra")
        uploaded_files = st.file_uploader("Choose spectrum files (.txt)", type="txt", accept_multiple_files=True)
        
        # Parameters
        st.subheader("Analysis Parameters")
        knn_neighbors = st.slider("Number of KNN Neighbors", min_value=1, max_value=20, value=5)
        
        # Debug options
        st.subheader("Debug Options")
        enable_debug = st.checkbox("Enable Debug Mode", value=True)
        
        # Process button
        process_btn = st.button("Process Spectra", type="primary")
    
    # Main content
    if uploaded_model and uploaded_files and process_btn:
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(uploaded_model)
        
        if model:
            st.success(f"Model loaded successfully! Training samples: {model.get('sample_size', 'Unknown')}")
            
            # Show model info
            if enable_debug:
                st.subheader("Model Information")
                st.write(f"- Number of components: {model.get('n_components', 'Unknown')}")
                st.write(f"- Variance threshold: {model.get('variance_threshold', 'Unknown')}")
                st.write(f"- Reference frequencies shape: {model['reference_frequencies'].shape}")
                st.write(f"- Training formulas: {np.unique(model['formulas'])}")
            
            # Process spectra
            with st.spinner("Processing spectra..."):
                results = process_spectra(model, uploaded_files, knn_neighbors)
            
            if results:
                results['uploaded_files'] = uploaded_files
                st.success(f"Successfully processed {len(results['new_embeddings'])} spectra!")
                
                # Create visualizations
                create_visualizations(results)
    
    elif not uploaded_model and process_btn:
        st.error("Please upload a trained model file (.pkl)")
    
    elif not uploaded_files and process_btn:
        st.error("Please upload at least one spectrum file (.txt)")
    
    else:
        # Show instructions
        st.info("""
        ### Instructions:
        1. **Upload a trained model** (.pkl file) in the sidebar
        2. **Upload one or more spectrum files** (.txt format) to analyze
        3. **Adjust parameters** like the number of KNN neighbors
        4. **Click 'Process Spectra'** to run the analysis
        
        ### Expected Input Format:
        Spectrum files should be in .txt format with either:
        - A header line containing parameters (e.g., `// molecules='CH3OH' logn=13.5 tex=150.0 velo=5.0 fwhm=1.0`)
        - Or just frequency-intensity pairs without header
        
        ### Output:
        The app will generate:
        - PCA variance and component plots
        - UMAP visualizations colored by parameters and molecular formula
        - K-Nearest Neighbors analysis
        - Individual prediction details
        - Downloadable results in CSV and pickle formats
        """)

if __name__ == "__main__":
    main()
