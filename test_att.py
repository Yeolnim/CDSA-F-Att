# coding=UTF-8

import pandas as pd
import jieba
import numpy as np
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        outputs = K.permute_dimensions(a * x, (0, 2, 1))
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

a=AttentionLayer()

model = load_model('./models/cdsa_f_att/dvd+electronics.h5',custom_objects={'AttentionLayer': a})
train = pd.read_csv('./data/dvd+electronics.csv')
test = pd.read_csv('./data/dvd.csv')

train['sen_cut'] = train['comment'].apply(jieba.lcut)
test['comment'] = test['comment'].astype(str)
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

# print(type(X_test))
# print(X_test.shape)
# print(X_test)

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
