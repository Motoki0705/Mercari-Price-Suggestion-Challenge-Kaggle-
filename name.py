import sys
import os

# 実行中のファイルが置かれているディレクトリを取得
sys.path.append(os.path.dirname(__file__))

from Encoder.Encoder_model import Encoder

def name_layer():
    d_model = 20
    embedding_dim = 128
    head_num = 8
    dropout_rate = 0.2
    n = 6
    name='name_inputs'
    
    inputs_text, inputs_mask, outputs = Encoder(d_model=d_model, embedding_dim=embedding_dim, head_num=head_num, dropout_rate=dropout_rate, N=n, input_name=name)
    
    return inputs_text, inputs_mask, outputs
    
