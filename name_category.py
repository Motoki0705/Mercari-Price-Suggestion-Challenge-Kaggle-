from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, BatchNormalization, Embedding

def name_category_layer(training=True):
    # ハイパーパラメータ
    vocab_size = 1000  # 埋め込みを作成するカテゴリの数（カテゴリの最大値を考慮）
    embedding_dim = 64  # 埋め込みベクトルの次元数
    input_length = 6   # 各入力に含まれるカテゴリ数（6つのカテゴリ）
    # LSTM層を1層に減らしてみる
    inputs = Input(shape=(6, ), name='name_category_inputs')
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length, mask_zero=True)(inputs)
    x = LSTM(units=64)(x)
    x = Dense(units=64*4, activation='relu')(x)
    x = BatchNormalization()(x, training=training)
    outputs = Dense(units=64, activation='relu')(x)
    
    return inputs, outputs



