import streamlit as st
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import re
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import tempfile
import base64
from io import BytesIO

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
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem;}
    .section-header {font-size: 1.8rem; color: #ff7f0e; margin-top: 2rem; margin-bottom: 1rem;}
    .info-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .stButton>button {width: 100%;}
    .plot-container {border: 1px solid #ddd; border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)

def sanitize_filename(filename):
    """Elimina caracteres inválidos de los nombres de archivo"""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def load_model(uploaded_file):
    """Carga el modelo entrenado desde un archivo subido"""
    try:
        model = pickle.load(uploaded_file)
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

def load_and_interpolate_spectrum(file_content, reference_frequencies, filename):
    """Carga un espectro .txt y lo interpola a las frecuencias de referencia"""
    lines = file_content.decode('utf-8').splitlines()
    
    # Determinar el formato del archivo
    first_line = lines[0].strip()
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
        except Exception as e:
            st.warning(f"Could not parse line '{line}': {e}")
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

    return spectrum_data, interpolated, formula, params, filename

def plot_example_spectrum(spectrum_data, interpolated, reference_frequencies, formula, filename):
    """Grafica el espectro original vs interpolado"""
    fig, ax = plt.subplots(figsize=(10, 5))
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

def plot_pca_cumulative_variance(cumulative_variance, variance_threshold, n_components):
    """Plot PCA cumulative explained variance"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-o')
    ax.axhline(y=variance_threshold, color='r', linestyle='--', label=f'{variance_threshold*100:.1f}% variance')
    ax.axvline(x=n_components, color='r', linestyle='--', label=f'{n_components} components')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Cumulative Explained Variance')
    ax.grid(True)
    ax.legend()
    return fig

def plot_pca_components(pca_components, reference_frequencies, n_components):
    """Plot first few PCA components"""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(min(5, n_components)):
        ax.plot(reference_frequencies, pca_components[i], label=f'PC {i+1}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Component Value')
    ax.set_title('First 5 Principal Components')
    ax.legend()
    ax.grid(True)
    return fig

def plot_umap_predictions(embedding_train, y_train, embedding_new, y_new, formulas_new, param_names, param_labels):
    """Plot UMAP with training and prediction data colored by parameters"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for i, (ax, param_name, param_label) in enumerate(zip(axes, param_names, param_labels)):
        # Plot training data
        param_values_train = y_train[:, i]
        sc_train = ax.scatter(embedding_train[:, 0], embedding_train[:, 1], 
                             c=param_values_train, cmap='viridis', alpha=0.4, s=8, label='Training')

        # Plot new predictions
        param_values_new = y_new[:, i]
        sc_new = ax.scatter(embedding_new[:, 0], embedding_new[:, 1], 
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
        for j in range(len(embedding_new)):
            ax.annotate(formulas_new[j], (embedding_new[j, 0], embedding_new[j, 1]),
                       fontsize=8, alpha=0.9, xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_umap_formulas(embedding_train, formulas_train, embedding_new, formulas_new):
    """Plot UMAP colored by molecular formula"""
    all_formulas = np.concatenate([formulas_train, formulas_new])
    unique_formulas = np.unique(all_formulas)

    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_formulas)))
    formula_to_color = {formula: color for formula, color in zip(unique_formulas, colors)}

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot training data
    for formula in np.unique(formulas_train):
        mask = formulas_train == formula
        color = formula_to_color[formula]
        ax.scatter(embedding_train[mask, 0], embedding_train[mask, 1], 
                   color=color, alpha=0.4, s=15, label=f'{formula} (Train)')

    # Plot new predictions with stars
    for formula in np.unique(formulas_new):
        mask = formulas_new == formula
        color = formula_to_color[formula]
        ax.scatter(embedding_new[mask, 0], embedding_new[mask, 1], 
                   color=color, marker='*', s=200, edgecolors='black', linewidth=2,
                   alpha=1.0, label=f'{formula} (New)')

    ax.set_title('UMAP: Molecular Formula (Training + Predictions)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_umap_knn(embedding_train, embedding_new, knn_indices, formulas_new, filenames_new):
    """Plot UMAP with KNN connections"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all training data
    ax.scatter(embedding_train[:, 0], embedding_train[:, 1], 
                c='lightgray', alpha=0.3, s=15, label='Training Data')
    
    # Plot KNN neighbors for each prediction
    for i, (new_embedding, indices) in enumerate(zip(embedding_new, knn_indices)):
        if indices:  # Solo si hay vecinos válidos
            # Plot neighbors
            ax.scatter(embedding_train[indices, 0], embedding_train[indices, 1],
                       c='blue', alpha=0.6, s=50, label='KNN Neighbors' if i == 0 else "")
            
            # Plot prediction
            ax.scatter(new_embedding[0], new_embedding[1], 
                       c='red', s=200, marker='*', edgecolors='black', 
                       linewidth=2, label=f'Prediction {i+1}' if i == 0 else "")
            
            # Connect prediction to neighbors
            for idx in indices:
                ax.plot([new_embedding[0], embedding_train[idx, 0]],
                         [new_embedding[1], embedding_train[idx, 1]],
                         'gray', alpha=0.3, linestyle='--')
    
    # Add labels for new predictions
    for i in range(len(embedding_new)):
        ax.annotate(filenames_new[i], (embedding_new[i, 0], embedding_new[i, 1]),
                   fontsize=8, alpha=0.9, xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    
    ax.set_title('UMAP: K-Nearest Neighbors Analysis')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_3d_umap(embedding_train, y_train, embedding_new, y_new, param_name, param_label):
    """Create 3D UMAP plot with parameter as Z-axis"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Get the parameter index
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_idx = param_names.index(param_name)
    
    # Training data
    param_values_train = y_train[:, param_idx]
    sc_train = ax.scatter(embedding_train[:, 0], embedding_train[:, 1], param_values_train,
                        c=param_values_train, cmap='viridis', alpha=0.4, s=20, depthshade=True, label='Training')

    # New predictions
    param_values_new = y_new[:, param_idx]
    sc_new = ax.scatter(embedding_new[:, 0], embedding_new[:, 1], param_values_new,
                      c=param_values_new, cmap='plasma', alpha=1.0, s=100, marker='X', 
                      edgecolors='red', linewidth=2, depthshade=True, label='New Predictions')

    cbar = plt.colorbar(sc_train, ax=ax, shrink=0.5)
    cbar.set_label(param_label)

    ax.set_title(f'3D UMAP: {param_name} as Z-axis (Training + Predictions)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel(param_label)
    ax.legend()

    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    
    return fig

def main():
    st.title("🧪 Molecular Spectrum Analyzer with PCA/UMAP")
    st.markdown("""
    This app analyzes molecular spectra using PCA and UMAP dimensionality reduction techniques.
    Upload a trained model and new spectrum files to visualize their projections.
    """)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("Upload Files")
        
        # Model upload
        uploaded_model = st.file_uploader("Upload Trained Model (.pkl)", type=['pkl'])
        
        # Spectrum files upload
        uploaded_spectra = st.file_uploader("Upload Spectrum Files (.txt)", type=['txt'], accept_multiple_files=True)
        
        # Parameters
        st.header("Parameters")
        knn_neighbors = st.slider("Number of KNN Neighbors", min_value=1, max_value=20, value=5)
        
        # Process button
        process_btn = st.button("Process Spectra", type="primary")
    
    # Main content
    if uploaded_model and uploaded_spectra and process_btn:
        with st.spinner("Loading model and processing spectra..."):
            # Load model
            model = load_model(uploaded_model)
            
            if model:
                st.success("Model loaded successfully!")
                
                # Display model info
                with st.expander("Model Information"):
                    st.write(f"Training samples: {model.get('sample_size', 'N/A')}")
                    st.write(f"Number of PCA components: {model.get('n_components', 'N/A')}")
                    st.write(f"Variance threshold: {model.get('variance_threshold', 'N/A')}")
                    st.write(f"Target length: {model.get('target_length', 'N/A')}")
                
                # Process spectra
                scaler = model['scaler']
                pca = model['pca']
                umap_model = model['umap']
                ref_freqs = model['reference_frequencies']
                
                new_spectra_data = []
                new_formulas = []
                new_params = []
                new_filenames = []
                new_embeddings = []
                new_pca_components = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_spectra):
                    status_text.text(f"Processing {uploaded_file.name} ({i+1}/{len(uploaded_spectra)})")
                    try:
                        spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                            uploaded_file.getvalue(), ref_freqs, uploaded_file.name
                        )
                        
                        # Transform the spectrum
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
                    
                    progress_bar.progress((i + 1) / len(uploaded_spectra))
                
                if not new_embeddings:
                    st.error("No valid spectra found for prediction")
                    return
                
                new_embeddings = np.array(new_embeddings)
                new_params = np.array(new_params)
                new_formulas = np.array(new_formulas)
                new_pca_components = np.array(new_pca_components)
                
                st.success(f"Successfully processed {len(new_embeddings)} spectra!")
                
                # Find KNN neighbors
                knn_indices = find_knn_neighbors(model['embedding'], new_embeddings, k=knn_neighbors)
                
                # Display results
                st.header("Results")
                
                # PCA Analysis
                st.subheader("PCA Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pca_var = plot_pca_cumulative_variance(
                        model['cumulative_variance'], 
                        model.get('variance_threshold', 0.95), 
                        model.get('n_components', 50)
                    )
                    st.pyplot(fig_pca_var)
                
                with col2:
                    fig_pca_comp = plot_pca_components(
                        model['pca'].components_, 
                        model['reference_frequencies'], 
                        model.get('n_components', 50)
                    )
                    st.pyplot(fig_pca_comp)
                
                # UMAP Visualizations
                st.subheader("UMAP Visualizations")
                
                # UMAP by parameters
                param_names = ['logn', 'tex', 'velo', 'fwhm']
                param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
                
                fig_umap_params = plot_umap_predictions(
                    model['embedding'], model['y'], 
                    new_embeddings, new_params, new_formulas,
                    param_names, param_labels
                )
                st.pyplot(fig_umap_params)
                
                # UMAP by formula
                fig_umap_formula = plot_umap_formulas(
                    model['embedding'], model['formulas'],
                    new_embeddings, new_formulas
                )
                st.pyplot(fig_umap_formula)
                
                # UMAP with KNN
                fig_umap_knn = plot_umap_knn(
                    model['embedding'], new_embeddings, 
                    knn_indices, new_formulas, new_filenames
                )
                st.pyplot(fig_umap_knn)
                
                # 3D UMAP plots
                st.subheader("3D UMAP Visualizations")
                param_option = st.selectbox("Select parameter for Z-axis", param_names)
                param_label = param_labels[param_names.index(param_option)]
                
                fig_3d = plot_3d_umap(
                    model['embedding'], model['y'],
                    new_embeddings, new_params,
                    param_option, param_label
                )
                st.pyplot(fig_3d)
                
                # Example spectrum plot
                st.subheader("Example Spectrum")
                if len(new_spectra_data) > 0:
                    spectrum_idx = st.selectbox("Select spectrum to display", range(len(new_spectra_data)), format_func=lambda x: new_filenames[x])
                    
                    spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                        uploaded_spectra[spectrum_idx].getvalue(), ref_freqs, uploaded_spectra[spectrum_idx].name
                    )
                    
                    fig_example = plot_example_spectrum(
                        spectrum_data, interpolated, ref_freqs, formula, filename
                    )
                    st.pyplot(fig_example)
                
                # Download results
                st.subheader("Download Results")
                
                # Create prediction coordinates DataFrame
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
                        'logn': new_params[i, 0],
                        'tex': new_params[i, 1],
                        'velo': new_params[i, 2],
                        'fwhm': new_params[i, 3],
                        'knn_neighbors_count': len(neighbor_info)
                    })
                
                df_coords = pd.DataFrame(prediction_coords)
                
                # Convert DataFrame to CSV for download
                csv = df_coords.to_csv(index=False)
                st.download_button(
                    label="Download Prediction Coordinates (CSV)",
                    data=csv,
                    file_name="prediction_coordinates.csv",
                    mime="text/csv"
                )
                
                # Display the DataFrame
                st.dataframe(df_coords)
                
            else:
                st.error("Failed to load the model. Please check the file format.")

if __name__ == "__main__":
    main()
