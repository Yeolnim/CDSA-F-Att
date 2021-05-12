# file: test_base.py
# author: Yue Jian
# Copyright (C) 2020. All Rights Reserved.

import pandas as pd
import numpy as np
import jieba

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score

model = load_model('./model/cdsa_b/*.h5')
train = pd.read_csv('./data/*.csv')
test = pd.read_csv('./data/*.csv')

train['sen_cut'] = train['comment'].astype(str).apply(jieba.lcut)
test['sen_cut'] = test['comment'].astype(str).apply(jieba.lcut)
X_train = train['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
X_test = test['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
text = np.array(X_train)

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

print(accuracy_score(Y_test,test_predicted))
