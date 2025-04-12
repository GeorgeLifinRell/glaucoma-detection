import argparse
import imageio
import io
import numpy as np
import os
import tempfile
import torch
import SimpleITK as sitk
import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
from skimage.transform import resize

from model_definition._2d_resnet_18 import build_resnet18_2d
from model_definition._3d_resnet_18 import ResNet3D
from model_definition._3d_densenet_121 import DenseNet3D
from model_definition._3d_cnn_encoder import CNNEncoder3D

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--resnet18_2d_model_path',
        default='models/2d_resnet18_glaucoma_model.keras', 
        type=str, help='Path to the 2D ResNet-18 model'
    )
    parser.add_argument(
        '--resnet18_3d_model_path',
        default='models/3d_resnet_18_glaucoma_model.pth',
        type=str, help='Path to the 3D ResNet-18 model'
    )
    parser.add_argument(
        '--densenet121_3d_model_path',
        default='models/3d_densenet_121_glaucoma_model.pth',
        type=str, help='Path to the 3D DenseNet-121 model'
    )
    parser.add_argument(
        '--cnn_encoder_3d_model_path',
        default='models/3d_cnn_encoder_glaucoma_model.pth', 
        type=str, help='Path to the 3D CNN Encoder model'
    )
    parser.add_argument(
        '--xgb_model_path',
        default='models/xgboost_model.json',
        type=str, help='Path to the XGBoost model'
    )
    
    return parser.parse_args()

def load_models(args):
    print('Loading model...')
    device = torch.device('cpu')

    # Instantiate the models
    resnet18_3d_model = ResNet3D(num_classes=2)
    densenet121_3d_model = DenseNet3D(num_classes=2)
    cnn_encoder_3d_model = CNNEncoder3D()

    resnet18_2d_model_path = args.resnet18_2d_model_path
    resnet18_3d_model_path = args.resnet18_3d_model_path
    cnn_encoder_3d_model_path = args.cnn_encoder_3d_model_path
    densenet121_3d_model_path = args.densenet121_3d_model_path
    
    # Load the Keras model as usual
    resnet18_2d_model = keras.models.load_model(resnet18_2d_model_path)

    # Load PyTorch model weights correctly without reassigning
    resnet18_3d_model.load_state_dict(torch.load(resnet18_3d_model_path, map_location=device))
    densenet121_3d_model.load_state_dict(torch.load(densenet121_3d_model_path, map_location=device))
    cnn_encoder_3d_model.load_state_dict(torch.load(cnn_encoder_3d_model_path, map_location=device))

    # Set all models to evaluation mode
    resnet18_3d_model.eval()
    densenet121_3d_model.eval()
    cnn_encoder_3d_model.eval()
    return resnet18_2d_model, resnet18_3d_model, densenet121_3d_model, cnn_encoder_3d_model

def load_xgboost_model(args):
    xgb_model_path = args.xgb_model_path
    xgb_model = xgb.Booster()
    xgb_model.load_model(xgb_model_path)
    return xgb_model

# def mha_to_numpy(mha_file):
#     """Convert MHA file to numpy array"""
#     with tempfile.NamedTemporaryFile(suffix='.mha', delete=False) as tmp_file:
#         tmp_file.write(mha_file.getbuffer())
#         tmp_file_path = tmp_file.name
    
#     # Read the MHA file
#     reader = sitk.ImageFileReader()
#     reader.SetFileName(tmp_file_path)
#     image = reader.Execute()
    
#     # Convert to numpy array
#     array_data = sitk.GetArrayFromImage(image)
    
#     # Clean up temp file
#     os.unlink(tmp_file_path)
    
#     return array_data


def mha_to_numpy_3d_and_2d(mha_file, target_2d_shape=(64, 64)):
    """
    Convert an MHA file to a 3D numpy volume and a 2D numpy array (central slice).

    Args:
        mha_file: The MHA file to process.
        target_2d_shape (tuple): The desired shape of the 2D array (default is (64, 64)).

    Returns:
        tuple: A tuple containing:
            - 3D numpy array representing the volume.
            - 2D numpy array representing the central slice resized to the target shape.
    """

    # Save uploaded file to a temporary .mha file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mha") as tmp:
        tmp.write(mha_file.read())
        tmp_path = tmp.name

    try:
        # Read the MHA file using SimpleITK
        reader = sitk.ImageFileReader()
        reader.SetFileName(tmp_path)
        image = reader.Execute()

        # Convert to a 3D numpy array
        volume_3d = sitk.GetArrayFromImage(image)

        # Extract the central slice
        central_slice_index = volume_3d.shape[0] // 2
        slice_2d = volume_3d[central_slice_index]

        # Resize the 2D slice to the target shape
        slice_2d_resized = resize(slice_2d, target_2d_shape, anti_aliasing=True)

        return volume_3d, slice_2d_resized

    finally:
        # Clean up temporary file
        os.remove(tmp_path)

def create_gif_from_volume(volume, temp_dir):
    """Create a GIF from 3D volume"""
    # Normalize volume for better visualization
    volume_norm = (volume - volume.min()) / (volume.max() - volume.min()) * 255
    volume_norm = volume_norm.astype(np.uint8)
    
    # Create frames for the GIF
    gif_path = os.path.join(temp_dir, "volume_visualization.gif")
    frames = []
    
    # Use the middle 80% of slices for better visualization
    start_idx = int(volume.shape[0] * 0.1)
    end_idx = int(volume.shape[0] * 0.9)
    
    for i in range(start_idx, end_idx):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(volume_norm[i], cmap='gray')
        ax.axis('off')
        
        # Save the plot to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        plt.close(fig)
    
    # Create GIF
    imageio.mimsave(gif_path, frames, duration=0.5)
    return gif_path


def predict_resnet18_2d(resnet18_2d_model, np_image):
    """
    Loads a 2D image from a .npy file, reshapes it to (64,64,1) if needed,
    and returns the prediction score from the Keras ResNet-18 model.
    """
    print(type(np_image))
    if np_image.shape != (64, 64, 1):
        try:
            np_image = np.reshape(np_image, (64, 64, 1))
        except ValueError:
            print("Cannot be reshaped!")
            return
    # plt.imshow(np_image)
    np_image = np.expand_dims(np_image, axis=0)
    prediction = resnet18_2d_model.predict(np_image)
    return torch.tensor(prediction)

def predict_resnet18_3d(resnet18_3d_model, np_image):
    """
    Loads a 3D volume from a .npy file, preprocesses it, and returns the probability
    from the 3D ResNet-18 PyTorch model (probability for class 1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18_3d_model.to(device)
    resnet18_3d_model.eval() 
    # image = np.load(image_path)
    if np_image.shape[-1] == 1:
        np_image = np_image.squeeze(-1)
    
    image_tensor = torch.tensor(np_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = resnet18_3d_model(image_tensor)
    return output


def predict_cnn_encoder_3d(cnn_encoder_3d_model, npy_image):
    """
    Loads a 3D volume from a .npy file and returns the probability
    from the 3D CNN Encoder PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if npy_image.ndim == 5:  # e.g., (1,64,64,64,1)
        npy_image = np.squeeze(npy_image, axis=-1)
    elif npy_image.ndim == 4:  # e.g., (64,64,64,1)
        npy_image = np.squeeze(npy_image, axis=-1)
        npy_image = np.expand_dims(npy_image, axis=0)
    elif npy_image.ndim == 3:  # e.g., (64,64,64)
        npy_image = np.expand_dims(npy_image, axis=0)
    else:
        raise ValueError(f"Unexpected data shape: {npy_image.shape}")
    
    # Convert to tensor and add batch dimension: (1, 1, D, H, W)
    input_tensor = torch.tensor(npy_image, dtype=torch.float32).unsqueeze(0).to(device)
    cnn_encoder_3d_model.to(device)
    cnn_encoder_3d_model.eval()
    with torch.no_grad():
        output = cnn_encoder_3d_model(input_tensor)
    return output


def predict_densenet121_3d(densenet121_3d_model, npy_image):
    """
    Loads a 3D volume from a .npy file and returns the probability
    from the 3D DenseNet121 PyTorch model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if npy_image.ndim == 5:
        npy_image = np.squeeze(npy_image, axis=-1)
    elif npy_image.ndim == 4:
        npy_image = np.squeeze(npy_image, axis=-1)
        npy_image = np.expand_dims(npy_image, axis=0)
    elif npy_image.ndim == 3:
        npy_image = np.expand_dims(npy_image, axis=0)
    else:
        raise ValueError(f"Unexpected data shape: {npy_image.shape}")
    
    input_tensor = torch.tensor(npy_image, dtype=torch.float32).unsqueeze(0).to(device)
    densenet121_3d_model.to(device)
    densenet121_3d_model.eval()
    with torch.no_grad():
        output = densenet121_3d_model(input_tensor)
    return output

def get_feature_tensor(image_index, category='normal'):
    """
    Extracts features from 2D and 3D models for the given image index and category.
    For 2D, the corresponding .npy file from the 2D dataset is used.
    For 3D, the corresponding .npy file from the 3D dataset is used.
    
    The function expects each model's prediction to be either:
      - A 2D tensor of shape (1,1), or
      - A 2D tensor of shape (1,2) from which the glaucoma probability (second column) is selected.
      
    All features are then concatenated into a final tensor of shape (4,1).
    
    Args:
        image_index (int): The index used to construct the file name.
        category (str): Either 'normal' or 'glaucoma'.
        
    Returns:
        torch.Tensor: A tensor of shape (4,1) containing the features.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define file paths based on category
    if category == 'normal':
        image_2d = f'/kaggle/input/glaucoma-2d/normal_2d/n_{image_index}.npy'
        image_3d = f'/kaggle/input/glaucoma-3d/normal_3d/normal_3d/n_{image_index}.npy'
    elif category == 'glaucoma':
        image_2d = f'/kaggle/input/glaucoma-2d/poag_2d/g_{image_index}.npy'
        image_3d = f'/kaggle/input/glaucoma-3d/poag_3d/poag_3d/g_{image_index}.npy'
    else:
        raise ValueError('Category undefined')
    
    feature_2d = predict_resnet18_2d(image_2d)
    feature_resnet3d = predict_resnet18_3d(image_3d)
    feature_densenet3d = predict_densenet121_3d(image_3d)
    feature_cnn_encoder3d = predict_cnn_encoder_3d(image_3d)
    
    # Ensure each feature is a tensor
    if not torch.is_tensor(feature_2d):
        feature_2d = torch.tensor(feature_2d)
    if not torch.is_tensor(feature_resnet3d):
        feature_resnet3d = torch.tensor(feature_resnet3d)
    if not torch.is_tensor(feature_densenet3d):
        feature_densenet3d = torch.tensor(feature_densenet3d)
    if not torch.is_tensor(feature_cnn_encoder3d):
        feature_cnn_encoder3d = torch.tensor(feature_cnn_encoder3d)
    
    def adjust_feature(feature):
        if feature.ndim == 2:
            if feature.shape[1] == 2:
                return feature[:, 1:2]
            elif feature.shape[1] == 1:
                return feature
            else:
                raise ValueError(f"Unexpected feature shape: {feature.shape}")
        else:
            # If not 2D, try to reshape to (1,1)
            return feature.view(1,1)
    
    feature_2d = adjust_feature(feature_2d).to(device)
    feature_resnet3d = adjust_feature(feature_resnet3d).to(device)
    feature_densenet3d = adjust_feature(feature_densenet3d).to(device)
    feature_cnn_encoder3d = adjust_feature(feature_cnn_encoder3d).to(device)
    
    # Now concatenate along dimension 0. All tensors should now be (1,1)
    feature_tensor = torch.cat([feature_2d, feature_resnet3d, feature_densenet3d, feature_cnn_encoder3d], dim=0)
    return feature_tensor

def save_features_dataset(category='normal'):
    features_dir = f'/kaggle/working/{category}_stacked_features'
    os.makedirs(features_dir, exist_ok=True)
    for image_index in range(1, 1525):
        print(f'processing image_index: {image_index}')
        feature_tensor = get_feature_tensor(image_index, category=category)
        torch.save(feature_tensor, os.path.join(features_dir, f'{category[0]}_{image_index}.pt'))

def get_dmatrix_and_np_features(np_2d_image, np_3d_image, models_2d_and_3d):

    device = torch.device("cpu")

    feature_2d = predict_resnet18_2d(models_2d_and_3d[0], np_2d_image)
    feature_resnet3d = predict_resnet18_3d(models_2d_and_3d[1], np_3d_image)
    feature_densenet3d = predict_densenet121_3d(models_2d_and_3d[2], np_3d_image)
    feature_cnn_encoder3d = predict_cnn_encoder_3d(models_2d_and_3d[3], np_3d_image)

    # Ensure each feature is a tensor
    if not torch.is_tensor(feature_2d):
        feature_2d = torch.tensor(feature_2d)
    if not torch.is_tensor(feature_resnet3d):
        feature_resnet3d = torch.tensor(feature_resnet3d)
    if not torch.is_tensor(feature_densenet3d):
        feature_densenet3d = torch.tensor(feature_densenet3d)
    if not torch.is_tensor(feature_cnn_encoder3d):
        feature_cnn_encoder3d = torch.tensor(feature_cnn_encoder3d)
    
    def adjust_feature(feature):
        if feature.ndim == 2:
            if feature.shape[1] == 2:
                return feature[:, 1:2]
            elif feature.shape[1] == 1:
                return feature
            else:
                raise ValueError(f"Unexpected feature shape: {feature.shape}")
        else:
            # If not 2D, try to reshape to (1,1)
            return feature.view(1,1)
    
    feature_2d = adjust_feature(feature_2d).to(device)
    feature_resnet3d = adjust_feature(feature_resnet3d).to(device)
    feature_densenet3d = adjust_feature(feature_densenet3d).to(device)
    feature_cnn_encoder3d = adjust_feature(feature_cnn_encoder3d).to(device)
    
    # Now concatenate along dimension 0. All tensors should now be (1,1)
    feature_tensor = torch.cat([feature_2d, feature_resnet3d, feature_densenet3d, feature_cnn_encoder3d], dim=0)
    np_feature = feature_tensor.cpu().numpy().reshape(1, -1)
    dmatrix_feature = xgb.DMatrix(np_feature)
    return dmatrix_feature, np_feature

def get_xgb_prediction(dmatrix_feature, xgb_model):
    prediction = xgb_model.predict(dmatrix_feature)
    return prediction