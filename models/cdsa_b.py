#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : Jian Yue
# @Time： 2020.03.26

import pandas as pd
import jieba
import numpy as np

model_path = 'kitchen+video.h5'
train = pd.read_csv('..data/kitchen+video.csv')
train['sen_cut'] = train['comment'].apply(jieba.lcut)

X_train = train['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
y_train = pd.get_dummies((np.asarray(train["label"])))

text = np.array(X_train)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

vocab_size = 30000
maxlen = 120

print("开始统计语料的词频信息...")
t = Tokenizer(vocab_size)
t.fit_on_texts(text)
word_index = t.word_index

print('完整的字典大小：', len(word_index))
print("开始序列化句子...")
X_train = t.texts_to_sequences(X_train)

print("开始对齐句子序列...")
X_train = pad_sequences(X_train, maxlen=maxlen, padding='post')
print("完成！")

import copy
# 移除低频词
small_word_index = copy.deepcopy(word_index)  # 防止原来的字典被改变
x = list(t.word_counts.items())
s = sorted(x, key=lambda p:p[1], reverse=True)

print("移除word_index字典中的低频词...")
for item in s[20000:]:
    small_word_index.pop(item[0])  # 对字典pop
print("完成！")

# 词嵌入
from gensim.models.word2vec import Word2Vec

# Word2Vec向量
# wv_model = Word2Vec.load(w2v_path)

from bert_serving.client import BertClient
bc = BertClient()

# 定义随机矩阵
embedding_matrix = np.random.uniform(size=(vocab_size+1,768))
print("构建embedding_matrix...")
for word, index in small_word_index.items():
    try:
        word_vector = bc.encode([word])
        embedding_matrix[index] = word_vector
        # print("Word: [", index, "]")
    except:
        print("Word: [",word,"] not in wvmodel! Use random embedding instead.")
print("完成！")
print("Embedding matrix shape:\n",embedding_matrix.shape)

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, GRU, LSTM, Activation, Dropout, Embedding, Bidirectional
from keras.layers import Multiply, Concatenate, Dot
from sklearn.metrics import f1_score

# BiLSTM
wv_dim = 768
n_timesteps = maxlen
inputs = Input(shape=(maxlen,))
embedding_sequences = Embedding(vocab_size+1, wv_dim, input_length=maxlen, weights=[embedding_matrix])(inputs)
lstm = Bidirectional(LSTM(128, return_sequences= False))(embedding_sequences)
l = Dense(128, activation="tanh")(lstm)
l = Dropout(0.5)(l)
l = Dense(2, activation="softmax")(l)
m = Model(inputs, l)
m.summary()
m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
m.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)
m.save(model_path)
