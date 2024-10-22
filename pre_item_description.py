import sys
import os

# 実行中のファイルが置かれているディレクトリを取得
sys.path.append(os.path.dirname(__file__))

from Encoder.WordEmbedding import wordtokenizer
def preprocess_item_description(df, d_model):
    df = df['item_description']
    df.fillna('')
    X = df
    
    X, vocab_size, mask = wordtokenizer(X, d_model)
    return {'text': X, 'mask': mask}
