import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import INPUT_SHAPE, NUM_CLASSES, SEED, seed_everything
from model import train_and_evaluate
from preprocess import dummy_batch

def build_backbone(input_shape: tuple = INPUT_SHAPE) -> keras.Model:
    # Instantiates the chosen pretrained backbone here
    # (e.g. InceptionV3, ResNet50V2, EfficientNetB0, or MobileNetV2 from tf.keras.applications)
    raise NotImplementedError("Module C owner will instantiate the pretrained backbone here.")

def _build_probe_backbone(input_shape: tuple = INPUT_SHAPE) -> keras.Model:
    # Lightweight dummy backbone for immediate pipeline verification without downloading ImageNet weights
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(inputs)
    return keras.Model(inputs=inputs, outputs=x, name="ProbeBackbone")

def build_transfer_model(
    base_model: keras.Model,
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dense_units: int = 256,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-4
) -> keras.Model:
    # Freeze pretrained feature extractor and attach a trainable classification head
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"{base_model.name}_Transfer")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def unfreeze_for_fine_tuning(
    model: keras.Model,
    base_model: keras.Model = None,
    unfreeze_from_layer: int = 0,
    fine_tune_learning_rate: float = 1e-5
) -> keras.Model:
    # Selectively unfreeze top layers of the base model for fine-tuning with a reduced learning rate
    if base_model is None:
        for layer in model.layers:
            if isinstance(layer, keras.Model):
                base_model = layer
                break

    if base_model is None:
        raise ValueError("No nested base model found in the provided transfer model.")

    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

if __name__ == "__main__":
    seed_everything(SEED)

    # 1. Load synthetic data to verify transfer learning flow
    X_train, y_train = dummy_batch(batch_size=64, modality="deep")
    X_val, y_val = dummy_batch(batch_size=32, modality="deep")

    # 2. Instantiate backbone (swap _build_probe_backbone for build_backbone with real backbone)
    backbone = _build_probe_backbone()

    # 3. Stage 1: Train classification head with frozen backbone
    transfer_model = build_transfer_model(
        base_model=backbone,
        dense_units=128,
        dropout_rate=0.3,
        learning_rate=1e-4
    )

    print("Testing Stage 1: Feature extraction with frozen backbone:")
    train_and_evaluate(
        model=transfer_model,
        model_name="Pretrained_Stage1_Probe",
        config_desc="frozen_backbone_head_dense128",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=2,
        batch_size=16
    )

    # 4. Stage 2 (Optional): Fine-tune top layers with lower learning rate
    print("\nTesting Stage 2: Fine-tuning unfreezing top layers:")
    fine_tuned_model = unfreeze_for_fine_tuning(
        model=transfer_model,
        base_model=backbone,
        unfreeze_from_layer=len(backbone.layers) // 2,
        fine_tune_learning_rate=1e-5
    )

    train_and_evaluate(
        model=fine_tuned_model,
        model_name="Pretrained_Stage2_Probe",
        config_desc="fine_tuned_top_layers_adam1e-5",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=1,
        batch_size=16
    )
