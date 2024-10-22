import sys
import os

# 実行中のファイルが置かれているディレクトリを取得
sys.path.append(os.path.dirname(__file__))

from Encoder.WordEmbedding import wordtokenizer
from joblib import Memory
import pandas as pd
def preprocess_name(df, d_model):
    df = df['name']
    df.fillna('')
    X = df
    
    X, vocab_size, mask = wordtokenizer(X, d_model)
    return {'text': X, 'mask': mask}

if __name__ == '__main__':
    memory = Memory("merukari/cachedir", verbose=0)
    @memory.cache
    def load_data(path, sep='\t'):
        return pd.read_csv(path, sep=sep)

    df = load_data(r'C:\Users\kamim\OneDrive\デスクトップ\kaggle\merukari\train.tsv')
    df = df[:1000]

    preprocess_name(df)