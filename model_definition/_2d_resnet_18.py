import tensorflow as tf
from tensorflow.keras import layers, Model

def build_resnet18_2d(input_shape=(64, 64, 1), num_classes=2):
    """ResNet-18 modified for 64x64 glaucoma detection"""
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # Residual Blocks 
    def basic_block(x, filters, stride=1):
        shortcut = x
        x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.ReLU()(x)
        return x

    filter_sizes = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]
    for filters, stride in zip(filter_sizes, strides):
        x = basic_block(x, filters, stride)
        x = basic_block(x, filters)

    # Global Average Pooling & Fully Connected Layer
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)  
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)  
    x = layers.Dropout(0.5)(x)  
    outputs = layers.Dense(num_classes, activation='softmax')(x)  

    model = Model(inputs, outputs, name="ResNet-18-Glaucoma")
    return model
