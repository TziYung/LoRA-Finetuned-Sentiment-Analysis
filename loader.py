import pandas as pd
import tensorflow as tf
import numpy as np

def process_data(table_path, tkzr,
    text_column = 0, label_column = -1,
    max_len = 300, train_ratio = 0.8
    ):
    table = pd.read_parquet(table_path).iloc[:100]
    text = table.iloc[:, text_column].to_list()
    text = tkzr(text, max_length = max_len, truncation = True, padding = True) 
    if label_column is None:
        data = tf.data.Dataset.from_tensor_slices(text)
    else:
        label = np.array(table.iloc[:, label_column].to_list())
        label = label.astype(np.int32)
        data = tf.data.Dataset.from_tensor_slices((text, label))
    data = data.shuffle(table.shape[0])
    amount = table.shape[0] * train_ratio
    amount = int(amount)
    return data.take(amount), data.skip(amount)


