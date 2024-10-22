import tensorflow as tf
import numpy as np
from transformers import BertTokenizer

def wordtokenizer(inputs: list, d_model: int):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    # トークナイザでトークン化
    tokenized_inputs = tokenizer.batch_encode_plus(
        inputs,
        max_length=d_model,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    return tokenized_inputs['input_ids'], vocab_size, tokenized_inputs['attention_mask']

class WordEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, embedding_dim: int,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 埋め込み層を作成
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    
    def call(self, inputs: tf.Tensor):
  
        # 埋め込みベクトルを取得
        vector_matrix = self.embedding_layer(inputs)
        
        # 埋め込みベクトルを返す
        return vector_matrix
    
    # 埋め込み行列を取得するメソッド
    def get_embedding_matrix(self):
        return self.embedding_layer.weights[0].numpy()

# テストコード
if __name__ == "__main__":
    # テスト用のダミー入力データ
    test_inputs = [
        "Hello, how are you?",
        "This is a test input for WordEmbedding layer."
    ]
    d_model = 10  # シーケンスの最大長
    embedding_dim = 16  # 埋め込み次元
    
    test_inputs, vocab_size, attention_masks = wordtokenizer(test_inputs, d_model)
    
    # WordEmbedding レイヤーのインスタンスを作成
    word_embedding_layer = WordEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    # call メソッドをテストして埋め込みベクトルを取得
    output_vectors= word_embedding_layer(test_inputs)
    
    # 結果を表示
    print("Input sentences:", test_inputs)
    print("Input sentences shape:", test_inputs.shape)
    print("Output vectors shape:", output_vectors.shape)
    print("Attention masks:", attention_masks)
    
    # 埋め込み行列の取得をテスト
    embedding_matrix = word_embedding_layer.get_embedding_matrix()
    print("Embedding matrix shape:", embedding_matrix.shape)
