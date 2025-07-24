from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Reshape,
    Dropout, LayerNormalization,
    GlobalAveragePooling1D, BatchNormalization,
    Conv1D, MultiHeadAttention
)
from tensorflow.keras.initializers import TruncatedNormal
from typing import Tuple
from config.hyperparameters import ModelParams


class FlightModelBuilder:
    """Builds and compiles the transformer-LSTM hybrid model"""

    @staticmethod
    def transformer_encoder(inputs, head_size: int, num_heads: int, ff_dim: int, dropout: float):
        """Creates a transformer encoder block"""
        # Normalization and attention
        x = LayerNormalization(epsilon=1e-6)(inputs)
        x = MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=dropout,
            kernel_initializer=TruncatedNormal(mean=0., stddev=0.01)
        )(x, x)
        res = x + inputs

        # Feed-forward network
        x = LayerNormalization(epsilon=1e-6)(res)
        x = Conv1D(
            filters=ff_dim,
            kernel_size=1,
            activation="relu",
            kernel_initializer=TruncatedNormal(mean=0., stddev=0.01)
        )(x)
        x = Conv1D(
            filters=inputs.shape[-1],
            kernel_size=1,
            kernel_initializer=TruncatedNormal(mean=0., stddev=0.01)
        )(x)
        return x + res

    def build_model(self, input_shape: Tuple[int, int], output_len: int, params: ModelParams) -> Model:
        """Construct the complete model architecture"""
        input_layer = Input(shape=input_shape)
        x = input_layer

        # Transformer blocks
        for _ in range(params.transformer_blocks):
            x = self.transformer_encoder(
                x,
                params.head_size,
                params.num_heads,
                params.ff_dim,
                dropout=params.dropout_range[1]  # Using max dropout for architecture
            )

        # LSTM processing
        x = GlobalAveragePooling1D(data_format="channels_first")(x)
        x = Reshape((-1, 1))(x)
        x = LSTM(units=params.lstm_units[1], activation='tanh')(x)  # Using max units for architecture

        # Dense layers
        for units in params.dense_units[:-1]:
            x = Dense(
                units=units,
                activation='relu',
                kernel_initializer=TruncatedNormal(mean=0., stddev=0.001)
            )(x)
            x = BatchNormalization()(x)

        # Output layer
        outputs = Dense(units=params.dense_units[-1])(x)

        return Model(input_layer, outputs)