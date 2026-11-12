"""Hybrid Gated Convolutional Network + Transformer Encoder model, copied
from the notebook (the model described in the manuscript).
"""

from tensorflow.keras.layers import (BatchNormalization, Concatenate, Conv1D,
                                     Dense, Dropout, GlobalAveragePooling1D,
                                     Input, LayerNormalization,
                                     MultiHeadAttention, Multiply)
from tensorflow.keras.models import Model


# Gated Convolution Network
def GatedConvNetwork(x, filters):
    f = Conv1D(filters, kernel_size=3, padding='same', activation='tanh')(x)
    g = Conv1D(filters, kernel_size=3, padding='same', activation='sigmoid')(x)
    return Multiply()([f, g])


# Transformer Encoder
def transformer_encoder(x, head_size=64, num_heads=2, ff_dim=128, dropout=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn)
    x = BatchNormalization()(x)
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dropout(dropout)(ff)
    ff_out = Dense(x.shape[-1])(ff)
    return LayerNormalization(epsilon=1e-6)(x + ff_out)


# Build Hybrid Model
def build_gcn_transformer_model(input_shape=(10, 50)):
    inputs = Input(shape=input_shape)
    gcu = GatedConvNetwork(inputs, filters=64)
    trans = transformer_encoder(inputs)
    fusion = Concatenate()([gcu, trans])
    pooled = GlobalAveragePooling1D()(fusion)
    dense = Dense(128, activation='relu')(pooled)
    drop = Dropout(0.3)(dense)
    output = Dense(1, activation='sigmoid')(drop)
    return Model(inputs, output, name="GCN_Transformer_Hybrid")
