from tensorflow.keras.layers import Dense, LSTM, Input, BatchNormalization, Embedding
import sys
import os
# 実行中のファイルが置かれているディレクトリを取得
sys.path.append(os.path.dirname(__file__))
from Encoder.Encoder_model import Encoder

def log_message(message):
    print(f"[INFO] {message}")

def brand_name_layer():
    input_dim = 5000
    output_dim = 64
    input_length = 1
    name = 'brand_name_inputs'
    inputs = Input(shape=(1,), name=name)
    x = Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length, mask_zero=True)(inputs)
    x = Dense(units=64*4, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(units=64, activation='relu')(x)
    
    log_message(f"Brand name layer created: {name}")
    return inputs, outputs, name

def item_condition_id_layer():
    name = 'item_condition_id_inputs'
    inputs = Input(shape=(1,), name=name)
    outputs = Dense(64, activation='relu')(inputs)  # 修正: Denseレイヤーが正しく実行されていなかった
    log_message(f"Item condition ID layer created: {name}")
    return inputs, outputs, name

def name_category_layer():
    vocab_size = 1000
    embedding_dim = 64
    input_length = 6
    name = 'name_category_inputs'
    inputs = Input(shape=(6,), name=name)
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length, mask_zero=True)(inputs)
    x = LSTM(units=64)(x)
    x = Dense(units=64*4, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(units=64, activation='relu')(x)
    
    log_message(f"Name category layer created: {name}")
    return inputs, outputs, name

def item_description_layer():
    d_model = 64
    embedding_dim = 128
    head_num = 8
    dropout_rate = 0.2
    n = 6
    name = 'item_description_inputs'
    name_text = f'{name}_text'
    name_mask = f'{name}_mask'
    inputs_text, inputs_mask, outputs = Encoder(d_model=d_model, embedding_dim=embedding_dim, head_num=head_num, dropout_rate=dropout_rate, N=n, input_name=name)
    
    log_message(f"Item description layer created: {name}")
    return inputs_text, inputs_mask, outputs, name_text, name_mask

def name_layer():
    d_model = 20
    embedding_dim = 128
    head_num = 8
    dropout_rate = 0.2
    n = 6
    name = 'name_inputs'
    name_text = f'{name}_text'
    name_mask = f'{name}_mask'
    
    inputs_text, inputs_mask, outputs = Encoder(d_model=d_model, embedding_dim=embedding_dim, head_num=head_num, dropout_rate=dropout_rate, N=n, input_name=name)
    
    log_message(f"Name layer created: {name}")
    return inputs_text, inputs_mask, outputs, name_text, name_mask

def shipping_layer():
    name = 'shipping_inputs'
    inputs = Input(shape=(1,), name=name)
    outputs = Dense(64, activation='relu')(inputs)  # activaiton -> activation に修正
    log_message(f"Shipping layer created: {name}")
    return inputs, outputs, name

# すべてのレイヤー関数をリストとして管理
layers = [
    brand_name_layer,
    name_layer,
    name_category_layer,
    item_condition_id_layer,
    item_description_layer,
    shipping_layer
]

def all_layers():
    inputs_dict = {}
    outputs_list = []
    total_layers = len(layers)  # 総レイヤー数を取得
    for i, layer in enumerate(layers):
        log_message(f"Processing layer {i+1}/{total_layers}...")
        result = layer()  # レイヤーを生成
        if len(result) == 5:  # テキストとマスクがある場合
            inputs_text, inputs_mask, outputs, name_text, name_mask = result
            inputs_dict[name_text] = inputs_text
            inputs_dict[name_mask] = inputs_mask
            outputs_list.append(outputs)
            log_message(f"Layer {name_text} and {name_mask} added")
        elif len(result) == 3:  # 一般的なレイヤー
            inputs, outputs, name = result
            inputs_dict[name] = inputs
            outputs_list.append(outputs)
            log_message(f"Layer {name} added")
        else:
            log_message(f"Unknown layer format for layer {i+1}/{total_layers}")

    return inputs_dict, outputs_list

# テスト用の呼び出し
inputs_dict, outputs_list = all_layers()
log_message("All layers processed.")
