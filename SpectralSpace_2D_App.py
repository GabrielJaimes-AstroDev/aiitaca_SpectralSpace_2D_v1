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

def load_model(uploaded_model):
    """Carga el modelo entrenado desde un archivo subido"""
    try:
        model = pickle.load(uploaded_model)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Traceback: {traceback.format_exc()}")
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

def process_spectra(model, uploaded_files, knn_neighbors=5):
    """Procesa los espectros cargados y genera resultados"""
    results = {}
    
    # Extraer componentes del modelo
    scaler = model['scaler']
    pca = model['pca']
    umap_model = model['umap']
    ref_freqs = model['reference_frequencies']
    pca_components = model['pca'].components_
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
        
        # Debug: inspect file first
        inspect_spectrum_file(uploaded_file, uploaded_file.name)
        
        try:
            spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                uploaded_file, ref_freqs, uploaded_file.name
            )
            
            if interpolated is not None:
                # Transformar el espectro
                X_scaled = scaler.transform([interpolated])
                X_pca = pca.transform(X_scaled)
                X_umap = umap_model.transform(X_pca)
                
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

def create_visualizations(results):
    """Crea visualizaciones interactivas de los resultados"""
    if not results:
        return
    
    new_embeddings = results['new_embeddings']
    new_params = results['new_params']
    new_formulas = results['new_formulas']
    new_filenames = results['new_filenames']
    knn_indices = results['knn_indices']
    model = results['model']
    
    # 1. PCA cumulative variance (del modelo)
    st.subheader("PCA Cumulative Explained Variance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(model['cumulative_variance'])+1), model['cumulative_variance'], 'b-o')
    ax.axhline(y=model['variance_threshold'], color='r', linestyle='--')
    ax.axvline(x=model['n_components'], color='r', linestyle='--')
    ax.set_xlabel('Número de componentes')
    ax.set_ylabel('Varianza acumulada')
    ax.set_title('PCA Cumulative Explained Variance')
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    
    # 2. PCA components (del modelo)
    st.subheader("First 5 Principal Components")
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(min(5, model['n_components'])):
        ax.plot(model['reference_frequencies'], model['pca'].components_[i], label=f'PC {i+1}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Component Value')
    ax.set_title('First 5 Principal Components')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
    
    # 3. UMAP colored by parameters (training + new predictions)
    st.subheader("UMAP Projection Colored by Parameters")
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()

    for i, (ax, param_name, param_label) in enumerate(zip(axes, param_names, param_labels)):
        # Plot training data
        param_values_train = model['y'][:, i]
        sc_train = ax.scatter(model['embedding'][:, 0], model['embedding'][:, 1], 
                             c=param_values_train, cmap='viridis', alpha=0.4, s=8, label='Training')

        # Plot new predictions
        param_values_new = new_params[:, i]
        sc_new = ax.scatter(new_embeddings[:, 0], new_embeddings[:, 1], 
                           c=param_values_new, cmap='plasma', alpha=1.0, s=100, 
                           marker='X', edgecolors='red', linewidth=2, label='New Predictions')

        cbar = plt.colorbar(sc_train, ax=ax)
        cbar.set_label(param_label)

        ax.set_title(f'UMAP: {param_name} (Training + Predictions)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True)
        ax.legend()

        # Add labels for new predictions
        for j in range(len(new_embeddings)):
            ax.annotate(new_formulas[j], (new_embeddings[j, 0], new_embeddings[j, 1]),
                       fontsize=8, alpha=0.9, xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    
    # 4. UMAP by formula (training + new predictions)
    st.subheader("UMAP Projection Colored by Molecular Formula")
    all_formulas = np.concatenate([model['formulas'], new_formulas])
    unique_formulas = np.unique(all_formulas)

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_formulas)))
    formula_to_color = {formula: color for formula, color in zip(unique_formulas, colors)}

    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot training data
    for formula in np.unique(model['formulas']):
        mask = model['formulas'] == formula
        color = formula_to_color[formula]
        ax.scatter(model['embedding'][mask, 0], model['embedding'][mask, 1], 
                   color=color, alpha=0.4, s=15, label=f'{formula} (Train)')

    # Plot new predictions with stars
    for formula in np.unique(new_formulas):
        mask = new_formulas == formula
        color = formula_to_color[formula]
        ax.scatter(new_embeddings[mask, 0], new_embeddings[mask, 1], 
                   color=color, marker='*', s=200, edgecolors='black', linewidth=2,
                   alpha=1.0, label=f'{formula} (New)')

    ax.set_title('UMAP: Molecular Formula (Training + Predictions)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    # 5. UMAP with KNN neighbors highlighted
    st.subheader("K-Nearest Neighbors Analysis")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot all training data
    ax.scatter(model['embedding'][:, 0], model['embedding'][:, 1], 
                c='lightgray', alpha=0.3, s=15, label='Training Data')
    
    # Plot KNN neighbors for each prediction
    for i, (new_embedding, indices) in enumerate(zip(new_embeddings, knn_indices)):
        if indices:  # Solo si hay vecinos válidos
            # Plot neighbors
            ax.scatter(model['embedding'][indices, 0], model['embedding'][indices, 1],
                       c='blue', alpha=0.6, s=50, label='KNN Neighbors' if i == 0 else "")
            
            # Plot prediction
            ax.scatter(new_embedding[0], new_embedding[1], 
                       c='red', s=200, marker='*', edgecolors='black', 
                       linewidth=2, label=f'Prediction {i+1}' if i == 0 else "")
            
            # Connect prediction to neighbors
            for idx in indices:
                ax.plot([new_embedding[0], model['embedding'][idx, 0]],
                         [new_embedding[1], model['embedding'][idx, 1]],
                         'gray', alpha=0.3, linestyle='--')
    
    ax.set_title('UMAP: K-Nearest Neighbors Analysis')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    
    # 6. Individual prediction plots for each new spectrum with KNN neighbors table
    st.subheader("Individual Predictions with KNN Neighbors")
    
    for i in range(len(new_embeddings)):
        with st.expander(f"Prediction {i+1}: {new_formulas[i]} ({new_filenames[i]})"):
            col1, col2 = st.columns(2)
            
            with col1:
                # UMAP plot
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Plot all training data in light gray
                ax.scatter(model['embedding'][:, 0], model['embedding'][:, 1], 
                           c='lightgray', alpha=0.3, s=10, label='Training Data')
                
                # Highlight KNN neighbors if available
                if i < len(knn_indices) and knn_indices[i]:
                    neighbor_indices = knn_indices[i]
                    neighbor_formulas = model['formulas'][neighbor_indices]
                    unique_neighbor_formulas = np.unique(neighbor_formulas)
                    
                    colors_neighbors = plt.cm.Set3(np.linspace(0, 1, len(unique_neighbor_formulas)))
                    formula_to_color_neighbors = {formula: color for formula, color in zip(unique_neighbor_formulas, colors_neighbors)}
                    
                    for formula in unique_neighbor_formulas:
                        formula_mask = neighbor_formulas == formula
                        formula_indices = np.array(neighbor_indices)[formula_mask]
                        ax.scatter(model['embedding'][formula_indices, 0], model['embedding'][formula_indices, 1],
                                   c=[formula_to_color_neighbors[formula]] * len(formula_indices),
                                   alpha=0.8, s=60, label=f'{formula} (Neighbor)', edgecolors='black', linewidth=1)
                
                # Highlight the specific prediction
                ax.scatter(new_embeddings[i, 0], new_embeddings[i, 1], 
                           c='red', s=200, marker='*', edgecolors='black', 
                           linewidth=2, label=f'Prediction: {new_formulas[i]}')
                
                ax.set_title(f'Spectrum Prediction: {new_formulas[i]} ({new_filenames[i]})')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                ax.grid(True, alpha=0.3)
                ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=2)
                
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                # KNN neighbors table
                if i < len(knn_indices) and knn_indices[i]:
                    neighbor_indices = knn_indices[i]
                    table_data = []
                    
                    # Calculate averages per molecule formula
                    formula_params = {}
                    for idx in neighbor_indices:
                        neighbor_formula = model['formulas'][idx]
                        neighbor_params = model['y'][idx]
                        
                        if neighbor_formula not in formula_params:
                            formula_params[neighbor_formula] = {'logn': [], 'tex': [], 'velo': [], 'fwhm': [], 'count': 0}
                        
                        formula_params[neighbor_formula]['logn'].append(neighbor_params[0])
                        formula_params[neighbor_formula]['tex'].append(neighbor_params[1])
                        formula_params[neighbor_formula]['velo'].append(neighbor_params[2])
                        formula_params[neighbor_formula]['fwhm'].append(neighbor_params[3])
                        formula_params[neighbor_formula]['count'] += 1
                        
                        table_data.append([
                            neighbor_formula,
                            f"{neighbor_params[0]:.2f}",
                            f"{neighbor_params[1]:.2f}",
                            f"{neighbor_params[2]:.2f}",
                            f"{neighbor_params[3]:.2f}"
                        ])
                    
                    # Add average rows for each molecule formula
                    for formula, params_dict in formula_params.items():
                        if params_dict['count'] > 1:  # Only add average if more than one sample
                            avg_logn = np.mean(params_dict['logn'])
                            avg_tex = np.mean(params_dict['tex'])
                            avg_velo = np.mean(params_dict['velo'])
                            avg_fwhm = np.mean(params_dict['fwhm'])
                            
                            table_data.append([
                                f"{formula} (AVG)",
                                f"{avg_logn:.2f}",
                                f"{avg_tex:.2f}",
                                f"{avg_velo:.2f}",
                                f"{avg_fwhm:.2f}"
                            ])
                    
                    # Display table
                    st.markdown("**K-Nearest Neighbors**")
                    df_table = pd.DataFrame(table_data, columns=['Formula', 'log(n)', 'T_ex (K)', 'Velocity', 'FWHM'])
                    st.dataframe(df_table, use_container_width=True)
                else:
                    st.info("No KNN neighbors found for this prediction")
    
    # 7. Example spectrum plots for each new spectrum
    st.subheader("Example Spectra")
    cols = st.columns(2)
    
    for i, uploaded_file in enumerate(results['uploaded_files']):
        with cols[i % 2]:
            fig = plot_example_spectrum(uploaded_file, model['reference_frequencies'], uploaded_file.name)
            if fig:
                st.pyplot(fig)
                plt.close(fig)
    
    # 8. Download results
    st.subheader("Download Results")
    
    # Create prediction coordinates
    prediction_coords = []
    for i in range(len(new_embeddings)):
        # Get KNN neighbors information
        neighbor_info = []
        if i < len(knn_indices) and knn_indices[i]:
            for idx in knn_indices[i]:
                neighbor_info.append({
                    'formula': model['formulas'][idx],
                    'logn': model['y'][idx, 0],
                    'tex': model['y'][idx, 1],
                    'velo': model['y'][idx, 2],
                    'fwhm': model['y'][idx, 3],
                    'umap_x': model['embedding'][idx, 0],
                    'umap_y': model['embedding'][idx, 1]
                })
        
        prediction_coords.append({
            'filename': new_filenames[i],
            'formula': new_formulas[i],
            'umap_x': new_embeddings[i, 0],
            'umap_y': new_embeddings[i, 1],
            'knn_neighbors': neighbor_info,
            'parameters': {
                'logn': new_params[i, 0],
                'tex': new_params[i, 1],
                'velo': new_params[i, 2],
                'fwhm': new_params[i, 3]
            }
        })
    
    # Create a flattened version for CSV
    flattened_coords = []
    for pred in prediction_coords:
        flat_pred = {
            'filename': pred['filename'],
            'formula': pred['formula'],
            'umap_x': pred['umap_x'],
            'umap_y': pred['umap_y'],
            'logn': pred['parameters']['logn'],
            'tex': pred['parameters']['tex'],
            'velo': pred['parameters']['velo'],
            'fwhm': pred['parameters']['fwhm']
        }
        
        # Add neighbor information
        for j, neighbor in enumerate(pred['knn_neighbors']):
            flat_pred[f'neighbor_{j+1}_formula'] = neighbor['formula']
            flat_pred[f'neighbor_{j+1}_logn'] = neighbor['logn']
            flat_pred[f'neighbor_{j+1}_tex'] = neighbor['tex']
            flat_pred[f'neighbor_{j+1}_velo'] = neighbor['velo']
            flat_pred[f'neighbor_{j+1}_fwhm'] = neighbor['fwhm']
        
        flattened_coords.append(flat_pred)
    
    df_coords = pd.DataFrame(flattened_coords)
    
    # Download button for CSV
    csv = df_coords.to_csv(index=False)
    st.download_button(
        label="Download Prediction Coordinates (CSV)",
        data=csv,
        file_name="prediction_coordinates.csv",
        mime="text/csv"
    )
    
    # Download button for complete results
    predictions = {
        'X_new': np.array(results['new_spectra_data']),
        'y_new': new_params,
        'formulas_new': new_formulas,
        'filenames_new': new_filenames,
        'pca_components_new': new_pca_components,
        'umap_embedding_new': new_embeddings,
        'knn_neighbors': knn_indices,
        'model_info': {
            'model_path': "uploaded_model.pkl",
            'training_samples': model['sample_size'],
            'n_components': model['n_components']
        }
    }
    
    # Save to bytes buffer
    buffer = BytesIO()
    pickle.dump(predictions, buffer)
    buffer.seek(0)
    
    st.download_button(
        label="Download Complete Results (Pickle)",
        data=buffer,
        file_name="predictions_results.pkl",
        mime="application/octet-stream"
    )

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
            st.success(f"Model loaded successfully! Training samples: {model['sample_size']}")
            
            # Show model info
            if enable_debug:
                st.subheader("Model Information")
                st.write(f"- Number of components: {model['n_components']}")
                st.write(f"- Variance threshold: {model['variance_threshold']}")
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

