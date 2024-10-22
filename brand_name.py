from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization, Embedding

def brand_name_layer(training=True):
    input_dim = 5000
    output_dim = 64
    input_length = 1
    inputs = Input(shape=(1, ), name='brand_name_inputs')
    x = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length, mask_zero=True)(inputs)
    x = Dense(units=64*4, activation='relu')(x)
    x = BatchNormalization()(x, training=training)
    outputs = Dense(units=64, activation='relu')(x)
    
    return inputs, outputs

