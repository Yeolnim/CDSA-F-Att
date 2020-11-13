# coding=UTF-8

import pandas as pd
import jieba
import numpy as np
from keras.models import load_model

model = load_model('./models/cdsa_b/dvd+electronics.h5')
train = pd.read_csv('./data/dvd+electronics.csv')
test = pd.read_csv('./data/dvd.csv')

train['sen_cut'] = train['comment'].apply(jieba.lcut)
test["comment"] = test['comment'].astype(str)
test['sen_cut'] = test['comment'].apply(jieba.lcut)
X_train = train['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
X_test = test['sen_cut'].apply(lambda x: ' '.join(x)).tolist()

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
X_test = t.texts_to_sequences(X_test)
print("开始对齐句子序列...")
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
print("完成！")

predicted = np.array(model.predict(X_test))
print(predicted)
test_predicted=np.argmax(predicted,axis=1)

Y_test = np.asarray(test["label"])

print(len(Y_test))
print(Y_test)
print(len(test_predicted))
print(test_predicted)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,test_predicted))
