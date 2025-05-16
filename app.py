import matplotlib.pyplot as plt
import tempfile
import streamlit as st
import numpy as np

from src.utils.utils import (
    parse_args,
    create_gif_from_volume,
    mha_to_numpy_3d_and_2d,
    load_models,
    load_xgboost_model,
    get_dmatrix_and_np_features,
    get_xgb_prediction,
    meta_predict
)


def create_probability_chart(prediction_score):
    """Create a bar chart showing the probabilities (1=glaucoma, 0=normal)"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    categories = ['Glaucoma', 'Normal']
    probabilities = [1 - prediction_score, prediction_score]  
    colors = ['red', 'green']
    
    bars = ax.bar(categories, probabilities, color=colors, alpha=0.7)
    

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Prediction Probabilities', fontsize=14)
    plt.tight_layout()
    
    return fig

def main():
    args = parse_args()
    st.title("Glaucoma Detection System")
    
    with st.spinner('Loading models... This might take a minute.'):
        models_2d_and_3d_in_order = load_models(args=args)
        xgb_model = load_xgboost_model(args=args)
    
    st.success("Models loaded successfully!")
    
    st.markdown("""
    ## Upload an PT/NPY File format for glaucoma detection.
    """)
    
    uploaded_file = st.file_uploader("Choose a file", type=["mha", "npy", "pt"])
    if uploaded_file is not None:
        with st.spinner('Processing the file...'):
            if uploaded_file.type == "mha":

                temp_dir = tempfile.mkdtemp()
                
                image_3d, image_2d = mha_to_numpy_3d_and_2d(uploaded_file)
                st.success("MHA file processed successfully!")
                
                st.write(f"3D Volume dimensions: {image_3d.shape}")
                st.write(f"2D image dimensions: {image_2d.shape}")
            
            st.subheader("Glaucoma Detection")
            with st.spinner('Analyzing the scan...'):
                prediction_score = None
                prediction_class = None
                
                if uploaded_file.name.split('.')[-1].lower() == "pt":
                    prediction_class, prediction_score = meta_predict(uploaded_file, xgb_model)
                    # Ensure prediction_score is between 0 and 1
                    prediction_score = float(prediction_score)
                else:
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
                    prediction_score = float(prediction_score[0])
                    prediction_class = "Glaucoma" if prediction_score > 0.5 else "Normal"
                # Display results
                st.markdown(f"### Prediction Result: **{('Normal' if prediction_class == 0 else 'Glaucoma')}**")
                st.markdown(f"**Probability of Normal Score:** {prediction_score:.4f}")
                st.markdown(f"**Probability of Glaucoma Score:** {1 - prediction_score:.4f}")
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                # with col1:
                #     #st.pyplot(create_prediction_gauge(prediction_score))
                #     pass
                
                with col2:
                    st.pyplot(create_probability_chart(prediction_score))
                

if __name__ == '__main__':
    main()