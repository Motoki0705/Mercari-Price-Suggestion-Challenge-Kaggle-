import sys
import os

# 実行中のファイルが置かれているディレクトリを取得
sys.path.append(os.path.dirname(__file__))

import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from joblib import Memory
from pre_brand_name import preprocess_brand_name
from pre_item_description import preprocess_item_description
from pre_name import preprocess_name
from pre_name_category import preprocess_name_category
from brand_name import brand_name_layer
from item_description import item_description_layer
from name import name_layer
from name_category import name_category_layer
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Lambda

# キャッシュ機能を使用してデータのロードを高速化
memory = Memory("merukari/cachedir", verbose=0)

# データをキャッシュして読み込む関数
@memory.cache
def load_data(path, sep='\t'):
    return pd.read_csv(path, sep=sep)

# データの読み込みとサンプルの制限
df = load_data(r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\train.tsv')
df = df[:1000]  # データサイズを1000に制限

class AdjustDim(tf.keras.layers.Layer):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis
    
    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            # Lambdaレイヤーでマスクにもsqueezeを適用する
            return Lambda(lambda m: tf.squeeze(m, axis=self.axis))(mask)
        return mask


# モデルの定義
def Model():
    
    adjust_dim = AdjustDim(1)  # 1次元目を削除するカスタムレイヤーのインスタンス

    # 各データパーツに対応するレイヤーの呼び出し
    brand_name_inputs, brand_name_outputs = brand_name_layer()  # ブランド名のレイヤー
    item_description_inputs_text, item_description_inputs_mask, item_description_outputs = item_description_layer()  # 商品説明のレイヤー
    name_inputs_text, name_inputs_mask, name_outputs = name_layer()  # 商品名のレイヤー
    name_category_inputs, name_category_outputs = name_category_layer()  # 商品カテゴリのレイヤー

    # 各出力をまとめ、必要に応じて次元調整（(None, 1, d_model)の形を(None, d_model)に変換）
    outputs = [brand_name_outputs, item_description_outputs, name_outputs, name_category_outputs]
    adjusted_output = [adjust_dim(output) if len(output.shape) == 3 and output.shape[1] == 1 else output for output in outputs]
    
    # 調整した出力を結合
    concat_layer = concatenate(adjusted_output)
    # 隠れ層の定義
    x = Dense(128, activation='relu')(concat_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # 出力層（1つの数値を出力）
    outputs = Dense(1, activation='linear')(x)
    
    # モデルの入力定義
    inputs = {'brand_name_inputs': brand_name_inputs,
              'item_description_inputs_text': item_description_inputs_text,
              'item_description_inputs_mask': item_description_inputs_mask,
              'name_inputs_text': name_inputs_text,
              'name_inputs_mask': name_inputs_mask,
              'name_category_inputs': name_category_inputs}
    
    # Kerasモデルのインスタンス化
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model  # モデルを返す

# メイン処理
if 1:
    d_model_item = 64  # 商品説明のモデル次元
    d_model_name = 20  # 商品名のモデル次元
    
    # 各データを前処理
    brand_name_inputs = preprocess_brand_name(df)  # ブランド名の前処理
    item_description_inputs = preprocess_item_description(df, d_model_item)  # 商品説明の前処理
    name_inputs = preprocess_name(df, d_model_name)  # 商品名の前処理
    name_category_inputs = preprocess_name_category(df)  # 商品カテゴリの前処理

    # Xに各前処理されたデータをまとめる
    X = {'brand_name_inputs': brand_name_inputs,
         'item_description_inputs_text': item_description_inputs['text'],
         'item_description_inputs_mask': item_description_inputs['mask'],
         'name_inputs_text': name_inputs['text'],
         'name_inputs_mask': name_inputs['mask'],
         'name_category_inputs': name_category_inputs}

    # ターゲット変数（価格）の正規化
    y = df['price'].values.reshape(-1, 1)
    scalar = MinMaxScaler()
    y_scaled = scalar.fit_transform(y)  # 価格データを[0, 1]に正規化
    
    # Xの各要素を個別に分割
    x_train_dict = {}
    x_test_dict = {}

    # 各特徴量をtrain_test_splitで分割
    for key, value in X.items():
        if isinstance(value, tf.Tensor):  # TensorFlowのテンソルの場合、NumPyに変換
            value = value.numpy()
        x_train_dict[key], x_test_dict[key] = train_test_split(value, test_size=0.1, random_state=0)

    # ターゲット変数を分割
    y_train, y_test = train_test_split(y_scaled, test_size=0.1, random_state=0)
    
    # モデルを作成し
    model = Model()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.predict(x_train_dict)
    
    # モデルのトレーニング（必要に応じて）
    if 0:
        model.fit(x_train_dict, y_train, epochs=100, batch_size=128, validation_split=0.2)
        model.save('model.keras')
