import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
import os
from tqdm import tqdm  # tqdm のインポート

# 実行中のファイルが置かれているディレクトリを取得
sys.path.append(os.path.dirname(__file__))
from Encoder.WordEmbedding import wordtokenizer
from joblib import Memory
import pandas as pd

# tqdmのpandas拡張を使用
tqdm.pandas()

def preprocess_brand_name(df):
    df = df['brand_name']
    df = df.fillna('')

    all_category = df.explode().unique()

    label_encoder = LabelEncoder()
    label_encoder.fit(all_category)

    def label_and_pad(x):
        if x == 'NaN':
            return 0
        else:
            return label_encoder.transform([x]) + 1
        
    X = np.array(df.apply(label_and_pad).tolist())
    name = 'brand_name_inputs'
    return X, name

def preprocess_item_condition(df):
    df = df['item_condition_id']
    df = df.fillna(0)
    X = df.to_numpy().astype('float32')
    name = 'item_condition_id_inputs'
    return X, name

def preprocess_item_description(df):
    df = df['item_description']
    df.fillna('')
    X = df
    
    X, vocab_size, mask = wordtokenizer(X, d_model=64)
    name_text = 'item_description_inputs_text'
    name_mask = 'item_description_inputs_mask'
    return {'text': X, 'mask': mask}, name_text, name_mask

def preprocess_name_category(df):

    # カテゴリの分解
    df['category_split'] = df['category_name'].str.split('/')
    
    for i in range(3):
        df[f'category_level_{i+1}'] = df['category_split'].progress_apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else 'unknown')

    for i in range(1, 4):
        df[f'category_level_{i}_split'] = df[f'category_level_{i}'].str.split('&').fillna('unknown')
        print(f'category_level_{i} split & replaced empty with "unknown"')
        
    df['category_split_all'] = df['category_name'].str.split(r'[/&]').fillna('unknown')

    # LabelEncoderの準備
    label_encoder = LabelEncoder()
    all_categories_unique = df['category_split_all'].explode().unique()

    # ユニークなカテゴリに基づいてフィット
    label_encoder.fit(all_categories_unique)

    # パディングとエンコードを行う関数
    def label_and_pad(x):
        if x == 'unknown':
            return (0, 0)  # 'unknown' の場合、(0, 0) としてエンコード
        elif len(x) == 1:
            return (label_encoder.transform(x)[0] + 1, 0) # 要素が1つなら (エンコード結果 + 1, 0)
        elif len(x) == 2:
            tmp = label_encoder.transform(x) + 1
            return (tmp[0], tmp[1])  # 要素が2つなら両方をエンコード

    # 各階層レベルの処理
    for i in range(1, 4):
        df[f'category_level_{i}_split_labeled'] = df[f'category_level_{i}_split'].apply(label_and_pad)
        
    # 横方向（列方向）に結合するために axis=1 を指定
    df['category_level_split'] = pd.concat([df['category_level_1_split_labeled'], 
                                            df['category_level_2_split_labeled'], 
                                            df['category_level_3_split_labeled']], axis=1).apply(tuple, axis=1)

    df['category_level_split'] = df['category_level_split'].apply(lambda x: [item for sublist in x for item in sublist])

    X = np.array(df['category_level_split'].tolist())
    name = 'name_category_inputs'
    return X, name

def preprocess_name(df):
    df = df['name']
    df.fillna('')
    X = df
    
    X, vocab_size, mask = wordtokenizer(X, d_model=20)
    name_text = 'name_inputs_text'
    name_mask = 'name_inputs_mask'
    return {'text': X, 'mask': mask}, name_text, name_mask
    
def preprocess_shipping(df):
    df = df['shipping']
    df = df.fillna(0)
    X = df.to_numpy().astype('float32')
    name = 'shipping_inputs'
    return X, name

preprocess_func_list = [
    preprocess_brand_name,
    preprocess_name,
    preprocess_name_category,
    preprocess_item_condition,
    preprocess_item_description,
    preprocess_shipping
]

def preprocess_all_columns(df):
    X = {}
    
    # tqdm を使って進捗を表示
    for func in tqdm(preprocess_func_list, desc="Preprocessing columns"):
        tmp = func(df)
        if len(tmp) == 3:
            dic, name_text, name_mask = tmp
            X[name_text] = dic['text']
            X[name_mask] = dic['mask']
        else:
            x, name = tmp
            X[name] = x
            
    return X
