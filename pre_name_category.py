import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Memory
import numpy as np
from tqdm import tqdm

# tqdmのpandas拡張を使用
tqdm.pandas()

memory = Memory("merukari/cachedir", verbose=0)

@memory.cache
def load_data(path, sep='\t'):
    return pd.read_csv(path, sep=sep)

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
    
    return X



def test_preprocess_name_category(df_test):

    # preprocess_name_category関数を呼び出して処理を行う
    X = preprocess_name_category(df_test)
    
    # テスト結果を表示
    print("Processed Category Data:\n", X)
    
    # 期待される出力形式の確認 (NumPy配列かどうか)
    assert isinstance(X, np.ndarray), "Output should be a NumPy array"
    print("Output is a NumPy array as expected.")
    
    # サンプルデータの一部を確認して、正しく処理されているかを手動でチェック
    expected_shape = (len(df_test), 6)  # 各カテゴリーは (label1, label2) 形式で3つのレベルがあるため
    assert X.shape == expected_shape, f"Output shape should be {expected_shape}, but got {X.shape}"
    print(f"Output shape is correct: {X.shape}")
    
    print("All tests passed successfully.")

if __name__ == '__main__':
        
    memory = Memory("merukari/cachedir", verbose=0)

    @memory.cache
    def load_data(path, sep='\t'):
        return pd.read_csv(path, sep=sep)

    df = load_data(r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\train.tsv')
    df = df[:1000]
    
    test_preprocess_name_category(df)
