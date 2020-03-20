#!/usr/bin/python
# _*_ coding: utf-8 _*_
import warnings
warnings.filterwarnings('ignore')
from keras import preprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models
from keras import layers
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import io
from sklearn.model_selection import train_test_split
import urllib
import urllib.parse
import io

# 加载数据
def load_data(file):
    with io.open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

x_normal = load_data("data/normal_parsed.txt")
x_anomalous = load_data("data/anomalous_parsed.txt")

# 创建数据集
x = x_normal  + x_anomalous

# 打标签
y_normal = [0] * len(x_normal)
y_anomalous = [1] * len(x_anomalous)
y = y_normal + y_anomalous

# 生成字符词典
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(x)

# 将字符按字符索引值转换为数值
sequences = tokenizer.texts_to_sequences(x)
char_index = tokenizer.word_index


print(char_index)

maxlen = 1000

# 填充序列至相同长度
x = pad_sequences(sequences, maxlen=maxlen)
y = np.asarray(y)

# 打乱数据
indices = np.arange(x.shape[0])
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)


# 创建验证集
x_val = x_train[:20000]
partial_x_train = x_train[20000:]
y_val = y_train[:20000]
partial_y_train = y_train[20000:]

# 向量空间大小
embedding_dim = 32

# 词汇大小
max_chars = 63


def build_model():
  model = models.Sequential()
  model.add(layers.Embedding(max_chars, embedding_dim, input_length=maxlen))
  model.add(layers.LSTM(100))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = build_model()
print(model.summary())

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=32)
test_acc, test_loss = model.evaluate(x_test, y_test)
print(test_acc, test_loss)


def test_it(file_in):
    fin = open(file_in)

    lines = fin.readlines()
    res = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("GET"):
            res.append("GET" + line.split(" ")[1])
        elif line.startswith("POST") or line.startswith("PUT"):
            url = line.split(' ')[0] + line.split(' ')[1]

            j = 1
            while True:
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 1
            data = lines[i + j + 1].strip()
            url += '?' + data
            res.append(url)
    for line in res:
        line = urllib.parse.unquote(line).replace('\n', '').lower()

    tokenizer = Tokenizer(char_level=True) 
    tokenizer.fit_on_texts(line)  # 将每行数据符号化

    sequences = tokenizer.texts_to_sequences(line)
    maxlen = 1000  # 最长输入序列长度

    x = pad_sequences(sequences, maxlen=maxlen)
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    my = model.predict(x)[0]

    if (my):
        print("input is a malicious web request")
    else:
        print("input is normal web request")

test_it('data/test.txt')

