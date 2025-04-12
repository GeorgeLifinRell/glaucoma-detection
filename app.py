import matplotlib.pyplot as plt
import tempfile
import streamlit as st

from src.utils.utils import (
    parse_args,
    create_gif_from_volume,
    mha_to_numpy_3d_and_2d,
    load_models,
    load_xgboost_model,
    get_dmatrix_and_np_features,
    get_xgb_prediction
)

def main():
    args = parse_args()
    st.title("Glaucoma Detection System")
    
    with st.spinner('Loading models... This might take a minute.'):
        models_2d_and_3d_in_order = load_models(args=args)
        xgb_model = load_xgboost_model(args=args)
    
    st.success("Models loaded successfully!")
    
    st.markdown("""
    ## Upload an MHA File
    Upload a 3D OCT scan in MHA format for glaucoma detection.
    """)
    
    uploaded_file = st.file_uploader("Choose an MHA file", type=["mha"])
    if uploaded_file is not None:
        with st.spinner('Processing the MHA file...'):
            # Create a temporary directory for our files
            temp_dir = tempfile.mkdtemp()
            
            # Process the MHA file
            image_3d, image_2d = mha_to_numpy_3d_and_2d(uploaded_file)
            st.success("MHA file processed successfully!")
            
            # Check if the volume data dimensions are suitable
            st.write(f"3D Volume dimensions: {image_3d.shape}")
            st.write(f"2D image dimensions: {image_2d.shape}")
            
            # Create visualization
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Visualizing the OCT Scan")
                gif_path = create_gif_from_volume(image_3d, temp_dir)
                st.image(gif_path, caption="3D Volume Visualization", use_container_width=True)
            
            # Extract features and make prediction
            with col2:
                st.subheader("Glaucoma Detection")
                with st.spinner('Analyzing the scan...'):
                    dmatrix_features, np_features = get_dmatrix_and_np_features(
                        image_2d,
                        image_3d,
                        models_2d_and_3d_in_order
                    )
                    
                    # Get prediction
                    prediction_score = get_xgb_prediction(
                        dmatrix_features,
                        xgb_model
                    )
                    prediction_score = prediction_score[0]
                    
                    # Display results
                    st.markdown(f"### Prediction Score: {prediction_score}")
                    
                    # Create a gauge chart for visualization
                    fig, ax = plt.subplots(figsize=(4, 4))
                    
                    # Create a simple gauge using a pie chart
                    colors = ['#3498db', '#e74c3c'] if prediction_score >= 0.5 else ['#e74c3c', '#3498db']
                    values = [prediction_score, 1-prediction_score] if prediction_score >= 0.5 else [1-prediction_score, prediction_score]
                    labels = ['Glaucoma', 'Normal'] if prediction_score >= 0.5 else ['Normal', 'Glaucoma']
                    
                    ax.pie(values, colors=colors, startangle=90, counterclock=False)
                    ax.add_artist(plt.Circle((0,0), 0.6, fc='white'))
                    plt.title('Confidence Score')
                    
                    # Add text in center
                    result = "Glaucoma Detected" if prediction_score >= 0.5 else "Normal"
                    confidence = prediction_score if prediction_score >= 0.5 else 1-prediction_score
                    ax.text(0, 0, f"{result}\n{confidence:.1%}", ha='center', va='center', fontsize=12)
                    
                    st.pyplot(fig)
                    
                    # Feature importance visualization
                    st.subheader("Model Features")
                    feature_names = ['2D ResNet', '3D ResNet', '3D DenseNet', '3D CNN']
                    feature_values = np_features.flatten()
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    bars = ax.barh(feature_names, feature_values, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
                    ax.set_xlabel('Feature Value')
                    ax.set_title('Individual Model Predictions')
                    
                    st.pyplot(fig)
                
            # Clean up temp directory
            import shutil
            shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
