import sys
import os

# 実行中のファイルが置かれているディレクトリを取得
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
import numpy as np
from WordEmbedding import WordEmbedding, wordtokenizer
from ResidualNormalizationWrapper import ResidualNormalizationWrapper
from AddPositionalEncording import AddPositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from FFN import FFN
from tensorflow.keras.layers import Input, concatenate
from joblib import Memory
import pandas as pd

class CreateMaskLayer(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def call(self, mask):
        # マスクを作成
        attention_mask = tf.tile(tf.expand_dims(mask, 2), [1, 1, self.d_model])
        return attention_mask
        
def Encoder(d_model: int, embedding_dim: int, head_num: int, dropout_rate: float, N: int, input_name: str, training=True):
    
    wordembedding = WordEmbedding(30522, embedding_dim)
    
    add_positional_encoding = AddPositionalEncoding()
    create_mask = CreateMaskLayer(d_model)

    # MultiHeadAttention and FFN with Residual Normalization Wrappers will be instantiated N times.
    M_ResidualNormalizationWappers = [
        ResidualNormalizationWrapper(MultiHeadAttention(embedding_dim, head_num), dropout_rate)
        for _ in range(N)
    ]
    F_ResidualNormalizationWappers = [
        ResidualNormalizationWrapper(FFN(embedding_dim, dropout_rate), dropout_rate)
        for _ in range(N)
    ]
    
    output_layer_1 = tf.keras.layers.Dense(int(embedding_dim*4), activation='relu')
    batch_norm = tf.keras.layers.BatchNormalization()
    output_layer_2 = tf.keras.layers.Dense(64, activation='relu')
            
    inputs_text = Input(shape=(d_model, ), name=f'{input_name}_text')
    inputs_mask = Input(shape=(d_model, ), name=f'{input_name}_mask')
    
    query= wordembedding(inputs_text) #[batch_size, d_model, embedding_dim]
    query = add_positional_encoding(query) #[batch_size, d_model, embedding_dim]
    attention_mask = create_mask(inputs_mask)

    # Instead of reusing the same Residual Normalization Wrappers, create new ones for each iteration.
    for i in range(N):
        query = M_ResidualNormalizationWappers[i](inputs=query, attention_mask=attention_mask, training=training) #[batch_size, d_model, embedding_dim]
        query = F_ResidualNormalizationWappers[i](inputs=query, training=training) #[batch_size, d_model, embedding_dim]
        
    x = query[:, -1, :] #[batch_size, embedding_dim]
    x = output_layer_1(x)
    x = batch_norm(x, training=training)
    outputs = output_layer_2(x)
    
    return inputs_text, inputs_mask, outputs

# テストコード
if __name__ == "__main__":
    # Encoder モデルのインスタンスを作成
    d_model = 10  # シーケンスの最大長
    embedding_dim = 16  # 埋め込み次元
    head_num = 4  # マルチヘッドアテンションのヘッド数
    dropout_rate = 0.1  # ドロップアウト率
    N = 2  # エンコーダの繰り返し数
    name_1 = 'inputs_layer_1'
    name_2 = 'inputs_layer_2'
    
    inputs_layer_1_text, inputs_layer_1_mask, outputs_1 = Encoder(d_model=d_model, embedding_dim=embedding_dim, head_num=head_num, dropout_rate=dropout_rate, N=N, input_name=name_1)
    inputs_layer_2_text, inputs_layer_2_mask, outputs_2 = Encoder(d_model=d_model, embedding_dim=embedding_dim, head_num=head_num, dropout_rate=dropout_rate, N=N, input_name=name_2)
    
    model = tf.keras.models.Model(inputs={
        'inputs_layer_1_text': inputs_layer_1_text,
        'inputs_layer_1_mask': inputs_layer_1_mask,
        'inputs_layer_2_text': inputs_layer_2_text,
        'inputs_layer_2_mask': inputs_layer_2_mask
    },
                                outputs=[outputs_1, outputs_2])
    
    memory = Memory("merukari/cachedir", verbose=0)
    @memory.cache
    def load_data(path, sep='\t'):
        return pd.read_csv(path, sep=sep)

    df = load_data(r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\train.tsv')
    df = df[:1000]

    inputs_name = preprocess_name(df, d_model)
    inputs_item_description = preprocess_item_description(df, d_model)
    
    inputs = {'inputs_layer_1_text': inputs_name['text'],
              'inputs_layer_1_mask': inputs_name['mask'],
              'inputs_layer_2_text': inputs_item_description['text'],
              'inputs_layer_2_mask': inputs_item_description['mask']}
    
    outputs = model.predict(inputs)
    
    # 結果を表示
    print("Encoder outputs:", outputs)
