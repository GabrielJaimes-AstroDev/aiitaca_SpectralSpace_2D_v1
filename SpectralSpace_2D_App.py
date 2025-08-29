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
    page_title="2D Spectral Space Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header2 {font-size: 2.5rem; color: #2ca02c; margin-bottom: 1rem;}
    .section-header {font-size: 1.8rem; color: #1f77b4; border-bottom: 2px solid #1f77b4; padding-bottom: 0.3rem; margin-top: 1.5rem;}
    .info-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .stButton>button {width: 100%;}
    .main-header {
        font-size: 1.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def sanitize_filename(filename):
    """Remove invalid characters from filenames"""
    invalid_chars = r'[<>:"/\\|?*]'
    return re.sub(invalid_chars, '_', filename)

def load_model(model_file):
    """Load trained model from uploaded file"""
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
    """Load spectrum from file content and interpolate to reference frequencies"""
    lines = file_content.decode('utf-8').splitlines()
    
    # Determine file format
    first_line = lines[0].strip()
    second_line = lines[1].strip() if len(lines) > 1 else ""
    
    formula = "Unknown"
    param_dict = {}
    data_start_line = 0
    
    # Format 1: with molecule and parameters header
    if first_line.startswith('//') and 'molecules=' in first_line:
        header = first_line[2:].strip()  # Remove the '//'
        formula = extract_molecule_formula(header)
        
        # Extract parameters from header
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
    
    # Format 2: with column header
    elif first_line.startswith('!') or first_line.startswith('#'):
        # Try to extract information from header if available
        if 'molecules=' in first_line:
            formula = extract_molecule_formula(first_line)
        data_start_line = 1
    
    # Format 3: no header, only data
    else:
        data_start_line = 0
        formula = filename.split('.')[0]  # Use filename as formula

    spectrum_data = []
    for line in lines[data_start_line:]:
        line = line.strip()
        # Skip comment or empty lines
        if not line or line.startswith('!') or line.startswith('#'):
            continue
            
        try:
            parts = line.split()
            if len(parts) >= 2:
                # Try different number formats
                try:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                except ValueError:
                    # Try scientific notation that might have D instead of E
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

    # Adjust frequency if in GHz (convert to Hz)
    if np.max(spectrum_data[:, 0]) < 1e11:  # If frequencies are less than 100 GHz, probably in GHz
        spectrum_data[:, 0] = spectrum_data[:, 0] * 1e9  # Convert GHz to Hz
        st.info(f"Converted frequencies from GHz to Hz for {filename}")

    interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1],
                            kind='linear', bounds_error=False, fill_value=0.0)
    interpolated = interpolator(reference_frequencies)

    # Extract parameters with default values if missing
    params = [
        param_dict.get('logn', np.nan),
        param_dict.get('tex', np.nan),
        param_dict.get('velo', np.nan),
        param_dict.get('fwhm', np.nan)
    ]

    return spectrum_data, interpolated, formula, params, filename

def find_knn_neighbors(training_embeddings, new_embeddings, k=5):
    """Find k nearest neighbors using KNN"""
    if len(training_embeddings) == 0 or len(new_embeddings) == 0:
        return []
    
    # Ensure k is not greater than number of training points
    k = min(k, len(training_embeddings))
    
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(training_embeddings)
    
    all_neighbor_indices = []
    for new_embedding in new_embeddings:
        distances, indices = knn.kneighbors([new_embedding])
        # Verify indices are within valid range
        valid_indices = [idx for idx in indices[0] if idx < len(training_embeddings)]
        all_neighbor_indices.append(valid_indices)
    
    return all_neighbor_indices

def truncate_filename(filename, max_length=20):
    """Truncate filename if it's too long for legend"""
    if len(filename) > max_length:
        return filename[:max_length-3] + "..."
    return filename

def truncate_title(title, max_length=50):
    """Truncate title if it's too long for plot"""
    if len(title) > max_length:
        return title[:max_length-3] + "..."
    return title

def main():

     # Add the header image and title
    st.image("NGC6523_BVO_2.jpg", use_column_width=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.empty()
        
    with col2:
        st.markdown('<p class="main-header">AI-ITACA | Artificial Intelligence Integral Tool for AstroChemical Analysis</p>', unsafe_allow_html=True)
    
    st.markdown("""
    A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.<br><br>
    <strong>Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)</strong>, proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>About GUAPOS</h4>
    <p>The G31.41+0.31 Unbiased ALMA sPectral Observational Survey (GUAPOS) project targets the hot molecular core (HMC) G31.41+0.31 (G31) to reveal the complex chemistry of one of the most chemically rich high-mass star-forming regions outside the Galactic center (GC).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header2">🧪 2D Spectral Space Analyzer</h1>', unsafe_allow_html=True)
    
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
            st.metric("Training Samples", model.get('sample_size', 'N/A'))
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
    st.subheader("A. UMAP Projection")
    
    # Create combined data for plotting
    train_df = pd.DataFrame({
        'umap_x': model['embedding'][:, 0],
        'umap_y': model['embedding'][:, 1],
        'formula': model['formulas'],
        'logn': model['y'][:, 0],
        'tex': model['y'][:, 1],
        'velo': model['y'][:, 2],
        'fwhm': model['y'][:, 3],
        'type': 'Predicted'
    })
    
    if len(results['umap_embedding_new']) > 0:
        # Create truncated filenames for legend
        truncated_filenames = [truncate_filename(fname) for fname in results['filenames_new']]
        
        new_df = pd.DataFrame({
            'umap_x': results['umap_embedding_new'][:, 0],
            'umap_y': results['umap_embedding_new'][:, 1],
            'formula': truncated_filenames,
            'full_filename': results['filenames_new'],
            'logn': results['y_new'][:, 0],
            'tex': results['y_new'][:, 1],
            'velo': results['y_new'][:, 2],
            'fwhm': results['y_new'][:, 3],
            'filename': results['filenames_new'],
            'type': 'New'
        })
        
        combined_df = pd.concat([train_df, new_df], ignore_index=True)
    else:
        combined_df = train_df
    
    # Create interactive UMAP plot with custom color sequence to avoid repetition
    # For training data (Predicted), we'll use a single color (black) and group them all as "Predicted"
    # For new data, we'll use different colors for each formula
    
    # Create a new column for coloring - for training data, use "Predicted", for new data use the formula
    combined_df['color_group'] = combined_df.apply(
        lambda row: 'Predicted' if row['type'] == 'Predicted' else row['formula'], 
        axis=1
    )
    
    # Get unique color groups and assign colors
    unique_groups = combined_df['color_group'].unique()
    color_map = {}
    
    # Assign black to all Predicted points
    color_map['Predicted'] = 'black'
    
    # Assign distinct colors to new spectra using a qualitative color scale
    new_groups = [group for group in unique_groups if group != 'Predicted']
    colors = px.colors.qualitative.Set3 + px.colors.qualitative.Set1 + px.colors.qualitative.Pastel
    
    for i, group in enumerate(new_groups):
        color_map[group] = colors[i % len(colors)]
    
    # Create the figure
    fig = px.scatter(combined_df, x='umap_x', y='umap_y', color='color_group',
                     symbol='type', 
                     hover_data=['logn', 'tex', 'velo', 'fwhm', 'full_filename' if 'full_filename' in combined_df.columns else 'filename'],
                     color_discrete_map=color_map)
    
    # Update layout to make it square and set colors/sizes
    fig.update_layout(
        width=700,
        height=700,
        autosize=False,
        legend=dict(
            itemsizing='constant',
            font=dict(size=10)
        )
    )
    
    # Update marker size and style for different types
    for i, trace in enumerate(fig.data):
        if trace.name == 'Predicted':
            trace.marker.update(size=10, symbol='circle')
        elif 'New' in trace.name:
            trace.marker.update(size=8, symbol='diamond')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Parameter distribution plots
    st.subheader("B. Parameter Distributions")
    
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    # Create subplots for each parameter
    fig = make_subplots(rows=2, cols=2, subplot_titles=param_labels)
    
    for i, param in enumerate(param_names):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        # Add training data with #2ca02c color
        fig.add_trace(
            go.Histogram(x=train_df[param], name='Predicted', opacity=0.7, marker_color='orange'),
            row=row, col=col
        )
        
        # Add new data if available
        if len(results['umap_embedding_new']) > 0:
            fig.add_trace(
                go.Histogram(x=new_df[param], name='New', opacity=0.7, marker_color='red'),
                row=row, col=col
            )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual spectrum analysis
    if len(results['umap_embedding_new']) > 0:
        st.subheader("C. Individual Spectrum Analysis")
        
        # Select a spectrum to analyze
        selected_idx = st.selectbox("Select a spectrum for detailed analysis", 
                                   range(len(results['filenames_new'])),
                                   format_func=lambda i: results['filenames_new'][i])
        
        if selected_idx is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Show interactive spectrum plot using Plotly
                st.markdown("**Spectrum Visualization**")
                
                # Create interactive plot with truncated title if needed
                truncated_title = truncate_title(f"Spectrum: {results['filenames_new'][selected_idx]}")
                
                spectrum_fig = go.Figure()
                spectrum_fig.add_trace(go.Scatter(
                    x=model['reference_frequencies'],
                    y=results['X_new'][selected_idx],
                    mode='lines',
                    name=truncate_filename(results['filenames_new'][selected_idx]),
                    line=dict(color='blue', width=2)
                ))
                
                spectrum_fig.update_layout(
                    title={
                        'text': truncated_title,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top',
                        'font': {
                            'size': 11
                        }
                    },
                    xaxis_title='Frequency (Hz)', 
                    yaxis_title='Intensity',
                    hovermode='x unified',
                    height=500,
                    width=600,
                    showlegend=False
                )
                
                # Add grid and other styling
                spectrum_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                spectrum_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                st.plotly_chart(spectrum_fig, use_container_width=True)
            
            with col2:
                # Calculate average parameters from KNN neighbors
                if 'knn_neighbors' in results and selected_idx < len(results['knn_neighbors']):
                    neighbor_indices = results['knn_neighbors'][selected_idx]
                    
                    if neighbor_indices:
                        # Calculate average parameters
                        avg_params = [
                            np.nanmean(model['y'][neighbor_indices, 0]),
                            np.nanmean(model['y'][neighbor_indices, 1]),
                            np.nanmean(model['y'][neighbor_indices, 2]),
                            np.nanmean(model['y'][neighbor_indices, 3])
                        ]
                        
                        # Find most common formula in neighbors
                        neighbor_formulas = [model['formulas'][idx] for idx in neighbor_indices]
                        most_common_formula = max(set(neighbor_formulas), key=neighbor_formulas.count)
                    else:
                        avg_params = [np.nan, np.nan, np.nan, np.nan]
                        most_common_formula = "Unknown"
                else:
                    avg_params = [np.nan, np.nan, np.nan, np.nan]
                    most_common_formula = "Unknown"
                
                # Show parameters
                st.markdown("**Estimated Parameters**")
                param_data = {
                    'Parameter': param_labels,
                    'Value': [
                        f"{avg_params[0]:.2f}" if not np.isnan(avg_params[0]) else "N/A",
                        f"{avg_params[1]:.2f}" if not np.isnan(avg_params[1]) else "N/A",
                        f"{avg_params[2]:.2f}" if not np.isnan(avg_params[2]) else "N/A",
                        f"{avg_params[3]:.2f}" if not np.isnan(avg_params[3]) else "N/A"
                    ]
                }
                st.table(pd.DataFrame(param_data))
                
                # Show molecule formula
                st.markdown(f"**Molecule Formula**: {most_common_formula}")
            
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
                    
                    # Create plot showing neighbors with hover information for all points
                    fig = go.Figure()
                    
                    # Add all training data with hover information
                    training_hover_text = [
                        f"Formula: {form}<br>log(n): {logn:.2f}<br>T_ex: {tex:.2f} K<br>Velocity: {velo:.2f} km/s<br>FWHM: {fwhm:.2f} km/s"
                        for form, logn, tex, velo, fwhm in zip(
                            train_df['formula'], train_df['logn'], train_df['tex'], 
                            train_df['velo'], train_df['fwhm']
                        )
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=train_df['umap_x'], y=train_df['umap_y'],
                        mode='markers',
                        marker=dict(color='lightgray', size=5),
                        name='Predicted Data',
                        text=training_hover_text,
                        hoverinfo='text'
                    ))
                    
                    # Add neighbors
                    neighbor_x = [model['embedding'][idx, 0] for idx in neighbor_indices]
                    neighbor_y = [model['embedding'][idx, 1] for idx in neighbor_indices]
                    neighbor_formulas = [model['formulas'][idx] for idx in neighbor_indices]
                    neighbor_logn = [model['y'][idx, 0] for idx in neighbor_indices]
                    neighbor_tex = [model['y'][idx, 1] for idx in neighbor_indices]
                    neighbor_velo = [model['y'][idx, 2] for idx in neighbor_indices]
                    neighbor_fwhm = [model['y'][idx, 3] for idx in neighbor_indices]
                    
                    neighbor_hover_text = [
                        f"Formula: {form}<br>log(n): {logn:.2f}<br>T_ex: {tex:.2f} K<br>Velocity: {velo:.2f} km/s<br>FWHM: {fwhm:.2f} km/s"
                        for form, logn, tex, velo, fwhm in zip(
                            neighbor_formulas, neighbor_logn, neighbor_tex, neighbor_velo, neighbor_fwhm
                        )
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=neighbor_x, y=neighbor_y,
                        mode='markers',
                        marker=dict(color='blue', size=10),
                        name='Neighbors',
                        text=neighbor_hover_text,
                        hoverinfo='text'
                    ))
                    
                    # Add selected spectrum
                    selected_hover_text = f"File: {results['filenames_new'][selected_idx]}<br>Formula: {most_common_formula}"
                    
                    fig.add_trace(go.Scatter(
                        x=[results['umap_embedding_new'][selected_idx, 0]],
                        y=[results['umap_embedding_new'][selected_idx, 1]],
                        mode='markers',
                        marker=dict(color='red', size=15, symbol='star'),
                        name='Selected Spectrum',
                        text=[selected_hover_text],
                        hoverinfo='text'
                    ))
                    
                    fig.update_layout(
                        title="K-Nearest Neighbors in UMAP Space",
                        width=700,
                        height=700,
                        autosize=False,
                        xaxis_title="UMAP 1",
                        yaxis_title="UMAP 2",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No neighbors found for this spectrum.")
            else:
                st.info("KNN analysis not available for this spectrum.")
    
    # Download results
    st.subheader("Download Results")
    
    if st.button("Export Results to CSV"):
        # Create results dataframe
        if len(results['umap_embedding_new']) > 0:
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
        else:
            st.warning("No results to export.")

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
    for spectrum_file in tqdm(spectra_files, desc="Processing spectra"):
        try:
            spectrum_data, interpolated, formula, params, filename = load_and_interpolate_spectrum(
                spectrum_file.getvalue(), spectrum_file.name, ref_freqs
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
            
        except Exception as e:
            st.warning(f"Error processing {spectrum_file.name}: {str(e)}")
            continue
    
    if not results['umap_embedding_new']:
        st.error("No valid spectra could be processed.")
        return results
    
    # Convert to arrays
    results['X_new'] = np.array(results['X_new'])
    results['y_new'] = np.array(results['y_new'])
    results['formulas_new'] = np.array(results['formulas_new'])
    results['umap_embedding_new'] = np.array(results['umap_embedding_new'])
    results['pca_components_new'] = np.array(results['pca_components_new'])
    
    # Find KNN neighbors
    results['knn_neighbors'] = find_knn_neighbors(
        model['embedding'], results['umap_embedding_new'], k=knn_neighbors
    )
    
    return results

if __name__ == "__main__":
    main()
