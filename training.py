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
import pickle
from preprocess_data import preprocess_all_columns
from layers import all_layers
from tensorflow.keras.layers import Dense, BatchNormalization, concatenate, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback

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
    
class PredictionCallback(Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        
    def on_epoch_end(self, epoch, logs=None):
        prediction = self.model.predict(self.x_test)
        idx = np.random.choice(100, 20, replace=False)
        df = pd.DataFrame({'pred': prediction.ravel(), 'label': self.y_test.ravel()})
        print(f'[{epoch+1}]  {df.iloc[idx]}')
        
# モデルの定義
def Model():
    
    adjust_dim = AdjustDim(1)  # 1次元目を削除するカスタムレイヤーのインスタンス

    inputs, outputs = all_layers()
    
    adjusted_output = [adjust_dim(output) if len(output.shape) == 3 and output.shape[1] == 1 else output for output in outputs]
    
    # 調整した出力を結合
    concat_layer = concatenate(adjusted_output)
    # 隠れ層の定義
    x = Dense(128, activation='relu')(concat_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    # 出力層（1つの数値を出力）
    outputs = Dense(1, activation='linear')(x)
    
    # Kerasモデルのインスタンス化
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    
    return model  # モデルを返す

# メイン処理
if 1:
    X = preprocess_all_columns(df)
    
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
    
    with open(r'merukari\code\data\train_data.pkl', 'wb') as f:
        pickle.dump((x_train_dict, x_test_dict, y_train, y_test), f)
        
        
with open(r'merukari\code\data\train_data.pkl', 'rb') as f:
    x_train_dict, x_test_dict, y_train, y_test = pickle.load(f)
    
# モデルを作成し
model = Model()
model.compile(optimizer='adam', loss='mean_squared_error')

# 学習率を減少させるコールバック
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
prediction_callback = PredictionCallback(x_test_dict, y_test)

# モデルのトレーニング（必要に応じて）
if 1:
    model.fit(x_train_dict, y_train, epochs=100, batch_size=128, validation_split=0.2, callbacks=[reduce_lr, prediction_callback])
    model.save('model.keras')
