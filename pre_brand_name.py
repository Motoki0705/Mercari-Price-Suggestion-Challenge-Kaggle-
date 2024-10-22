import numpy as np
from sklearn.preprocessing import LabelEncoder

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

    return X

