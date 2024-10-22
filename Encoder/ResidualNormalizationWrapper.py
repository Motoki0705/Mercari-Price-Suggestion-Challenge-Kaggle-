import tensorflow as tf
import numpy as np

class ResidualNormalizationWrapper(tf.keras.models.Model):
    def __init__(self, layer: tf.keras.layers.Layer, dropout_rate: float,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.layer = layer
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        
    def build(self, input_shape):
        self.layer_norm.build(input_shape)
        self.layer.build(input_shape)
        self.dropout_layer.build(input_shape)
        super().build(input_shape)
        
    def call(self, inputs: tf.Tensor, training: bool, *args, **kwargs):
        x = self.layer_norm(inputs)
        
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            x = self.layer(x, attention_mask=attention_mask, training=training)
        else:
            x = self.layer(x, training=training)
        x = self.dropout_layer(x, training=training)
        x += inputs
        return x
