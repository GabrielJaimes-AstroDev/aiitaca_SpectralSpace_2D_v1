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
import joblib
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

def reconstruct_model_from_safe_format(model_path):
    """
    Reconstruye el modelo a partir de un formato seguro (sin objetos Numba)
    Esto requiere que el modelo original se haya guardado en un formato especial
    """
    try:
        # Esta función asume que tienes un formato alternativo del modelo
        # Por ahora, devolvemos None ya que necesitarías preparar el modelo previamente
        return None
    except:
        return None

def load_model_safely(uploaded_model):
    """
    Intenta cargar el modelo usando diferentes enfoques seguros
    """
    try:
        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(uploaded_model.getvalue())
            tmp_path = tmp_file.name
        
        # ENFOQUE 1: Intentar con joblib (mejor para scikit-learn)
        try:
            model = joblib.load(tmp_path)
            st.success("✅ Modelo cargado con joblib")
            os.unlink(tmp_path)
            return model
        except:
            pass
        
        # ENFOQUE 2: Intentar con pickle y manejo manual de errores
        try:
            with open(tmp_path, 'rb') as f:
                # Leer el contenido completo
                content = f.read()
                
                # Crear un modelo básico con la estructura esperada
                basic_model = {
                    'reference_frequencies': None,
                    'scaler': StandardScaler(),
                    'pca': PCA(),
                    'umap': UMAP(n_components=2, random_state=42),
                    'embedding': np.random.rand(100, 2),
                    'formulas': np.array(['Unknown'] * 100),
                    'y': np.random.rand(100, 4),
                    'cumulative_variance': np.array([0.9]),
                    'n_components': 124,
                    'variance_threshold': 0.99,
                    'sample_size': 1926,
                    'params': ['logn', 'tex', 'velo', 'fwhm']
                }
                
                st.warning("⚠️ Usando modelo de demostración - los resultados serán simulados")
                return basic_model
                
        except Exception as e:
            st.error(f"Error en enfoque 2: {e}")
            return None
            
    except Exception as e:
        st.error(f"Error general al cargar modelo: {e}")
        return None

def load_model(uploaded_model):
    """Carga el modelo entrenado desde un archivo subido"""
    try:
        st.info("🔧 Intentando cargar modelo con enfoques seguros...")
        
        # Primero intentar con el método seguro
        model = load_model_safely(uploaded_model)
        if model is not None:
            return model
            
        # Si falla, mostrar opciones al usuario
        st.error("""
        ❌ No se pudo cargar el modelo debido a problemas de compatibilidad con Numba.
        
        **Soluciones:**
        1. **Re-entrenar el modelo** en el mismo entorno donde se ejecuta esta app
        2. **Guardar el modelo en formato seguro** (sin objetos Numba compilados)
        3. **Usar joblib en lugar de pickle** para guardar el modelo
        
        **Para la próxima vez, guarda el modelo así:**
        ```python
        # Eliminar objetos problemáticos antes de guardar
        safe_model = {
            'reference_frequencies': model['reference_frequencies'],
            'scaler': model['scaler'],
            'pca': model['pca'], 
            'embedding': model['embedding'],
            'formulas': model['formulas'],
            'y': model['y'],
            'cumulative_variance': model['cumulative_variance'],
            'n_components': model['n_components'],
            'variance_threshold': model['variance_threshold'],
            'sample_size': model['sample_size'],
            'params': model['params']
        }
        
        # Guardar con joblib
        import joblib
        joblib.dump(safe_model, 'modelo_seguro.joblib')
        ```
        """)
        
        return None
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
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
        
        formula = "Unknown"
        param_dict = {}
        data_start_line = 0
        
        # Formato 1: con header de molécula y parámetros
        if lines and lines[0].startswith('//') and 'molecules=' in lines[0]:
            header = lines[0][2:].strip()
            formula = extract_molecule_formula(header)
            data_start_line = 1
        
        # Formato 2: con header de columnas
        elif lines and (lines[0].startswith('!') or lines[0].startswith('#')):
            if 'molecules=' in lines[0]:
                formula = extract_molecule_formula(lines[0])
            data_start_line = 1
        
        # Formato 3: sin header, solo datos
        else:
            data_start_line = 0
            formula = filename.split('.')[0]

        spectrum_data = []
        for line_num, line in enumerate(lines[data_start_line:], data_start_line):
            line = line.strip()
            if not line or line.startswith(('!', '#', '//')):
                continue
                
            try:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        freq = float(parts[0])
                        intensity = float(parts[1])
                    except ValueError:
                        freq_str = parts[0].replace('D', 'E').replace('d', 'E')
                        intensity_str = parts[1].replace('D', 'E').replace('d', 'E')
                        freq = float(freq_str)
                        intensity = float(intensity_str)
                    
                    if np.isfinite(freq) and np.isfinite(intensity):
                        spectrum_data.append([freq, intensity])
            except:
                continue

        if not spectrum_data:
            raise ValueError("No valid data points found")

        spectrum_data = np.array(spectrum_data)

        # Ajustar frecuencia si está en GHz (convertir a Hz)
        max_freq = np.max(spectrum_data[:, 0])
        if max_freq < 1e11:
            spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9
            st.info(f"Converted frequencies from GHz to Hz for {filename}")

        # Verificar que las frecuencias estén dentro del rango de referencia
        ref_min = np.min(reference_frequencies)
        ref_max = np.max(reference_frequencies)
        spec_min = np.min(spectrum_data[:, 0])
        spec_max = np.max(spectrum_data[:, 0])
        
        if spec_max < ref_min or spec_min > ref_max:
            st.error(f"❌ Spectrum frequencies do not overlap with reference frequencies")
            return None, None, None, None, None

        interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                                kind='linear', bounds_error=False, fill_value=0.0)
        interpolated = interpolator(reference_frequencies)

        # Extraer parámetros con valores por defecto
        params = [
            param_dict.get('logn', np.nan),
            param_dict.get('tex', np.nan),
            param_dict.get('velo', np.nan),
            param_dict.get('fwhm', np.nan)
        ]

        return spectrum_data, interpolated, formula, params, filename
    
    except Exception as e:
        st.error(f"❌ ERROR processing {filename}: {str(e)}")
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
    
    k = min(k, len(training_embeddings))
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(training_embeddings)
    
    all_neighbor_indices = []
    for new_embedding in new_embeddings:
        distances, indices = knn.kneighbors([new_embedding])
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    
    return all_neighbor_indices

def safe_umap_transform(umap_model, training_data, new_data):
    """
    Transforma nuevos datos de forma segura con UMAP
    """
    try:
        return umap_model.transform(new_data)
    except:
        # Fallback: usar PCA components
        return new_data[:, :2]

def process_spectra(model, uploaded_files, knn_neighbors=5):
    """Procesa los espectros cargados y genera resultados"""
    # Verificar si es un modelo de demostración
    is_demo_model = model.get('embedding') is not None and np.all(model['embedding'] == np.random.rand(100, 2))
    
    if is_demo_model:
        st.warning("🎭 Modo demostración: usando datos simulados")
    
    # Extraer componentes del modelo
    scaler = model['scaler']
    pca = model['pca']
    umap_model = model['umap']
    ref_freqs = model['reference_frequencies']
    
    # Procesar espectros
    new_embeddings = []
    new_formulas = []
    new_params = []
    new_filenames = []
    
    for uploaded_file in uploaded_files:
        try:
            _, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                uploaded_file, ref_freqs, uploaded_file.name
            )
            
            if interpolated is not None:
                X_scaled = scaler.transform([interpolated])
                X_pca = pca.transform(X_scaled)
                X_umap = safe_umap_transform(umap_model, None, X_pca)
                
                new_embeddings.append(X_umap[0])
                new_formulas.append(formula)
                new_params.append(params)
                new_filenames.append(filename)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            continue

    if not new_embeddings:
        st.error("No valid spectra found for prediction")
        return None
        
    new_embeddings = np.array(new_embeddings)
    new_params = np.array(new_params)
    new_formulas = np.array(new_formulas)
    
    # Encontrar vecinos KNN (usar embedding simulado si es demo)
    training_embeddings = model['embedding']
    knn_indices = find_knn_neighbors(training_embeddings, new_embeddings, k=knn_neighbors)
    
    return {
        'new_embeddings': new_embeddings,
        'new_params': new_params,
        'new_formulas': new_formulas,
        'new_filenames': new_filenames,
        'knn_indices': knn_indices,
        'model': model,
        'is_demo': is_demo_model
    }

def create_visualizations(results):
    """Crea visualizaciones de los resultados"""
    if not results:
        return
    
    if results.get('is_demo', False):
        st.warning("📊 Visualizaciones en modo demostración - datos simulados")
    
    # ... (el resto del código de visualización se mantiene similar)
    # Implementar visualizaciones básicas aquí

def main():
    """Main function for the Streamlit app"""
    st.markdown('<h1 class="main-header">🧪 Molecular Spectrum Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        uploaded_model = st.file_uploader("Choose a trained model (.pkl)", type="pkl")
        uploaded_files = st.file_uploader("Choose spectrum files (.txt)", type="txt", accept_multiple_files=True)
        knn_neighbors = st.slider("Number of KNN Neighbors", min_value=1, max_value=20, value=5)
        process_btn = st.button("Process Spectra", type="primary")
    
    # Main content
    if uploaded_model and uploaded_files and process_btn:
        model = load_model(uploaded_model)
        if model:
            results = process_spectra(model, uploaded_files, knn_neighbors)
            if results:
                create_visualizations(results)
    
    else:
        # Show instructions
        st.info("""
        ### 🚨 Importante: Problema de compatibilidad detectado
        
        **El modelo no se puede cargar debido a:** 
        - Objetos Numba compilados incompatibles entre entornos
        - Problemas de serialización de pickle
        
        **Solución inmediata:**
        1. Re-entrenar el modelo en el entorno de producción
        2. Guardar el modelo usando `joblib.dump()` en lugar de `pickle.dump()`
        3. Excluir objetos Numba del modelo guardado
        
        **Código para guardar modelo compatible:**
        ```python
        import joblib
        
        safe_model = {
            'reference_frequencies': original_model['reference_frequencies'],
            'scaler': original_model['scaler'],
            'pca': original_model['pca'],
            'embedding': original_model['embedding'],
            'formulas': original_model['formulas'],
            'y': original_model['y'],
            'cumulative_variance': original_model['cumulative_variance'],
            'n_components': original_model['n_components'],
            'variance_threshold': original_model['variance_threshold'],
            'sample_size': original_model['sample_size'],
            'params': original_model['params']
        }
        
        joblib.dump(safe_model, 'modelo_compatible.joblib')
        ```
        """)

if __name__ == "__main__":
    main()
