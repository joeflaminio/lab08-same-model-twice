import keras
from keras import layers
import numpy as np
import torch


def build_keras_model(n_features: int, hidden: int = 16) -> keras.Model:
    inputs = keras.Input(shape=(n_features,), name="features")
    x = layers.Dense(hidden, activation="relu", name="hidden_1")(inputs)
    x = layers.Dense(hidden, activation="relu", name="hidden_2")(x)
    outputs = layers.Dense(1, activation=None, name="output")(x)
    squeezed = layers.Lambda(lambda t: keras.ops.squeeze(t, axis=-1), name="squeeze_output")(outputs)
    return keras.Model(inputs=inputs, outputs=squeezed, name="bike_mlp_keras")


def copy_weights_from_pytorch(keras_model: keras.Model, pt_model: torch.nn.Module) -> keras.Model:
    pt_linears = [m for m in pt_model.modules() if isinstance(m, torch.nn.Linear)]
    keras_denses = [
        layer for layer in keras_model.layers if isinstance(layer, layers.Dense)
    ]

    if len(pt_linears) != len(keras_denses):
        raise ValueError(
            f"Layer count mismatch: PyTorch has {len(pt_linears)} Linear layers, "
            f"Keras has {len(keras_denses)} Dense layers."
        )

    for pt_layer, k_layer in zip(pt_linears, keras_denses):
        w_np = pt_layer.weight.detach().cpu().numpy().T
        if pt_layer.bias is not None:
            b_np = pt_layer.bias.detach().cpu().numpy()
            k_layer.set_weights([w_np, b_np])
        else:
            k_layer.set_weights([w_np])

    return keras_model
