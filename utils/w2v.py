from gensim.models.word2vec import Word2Vec
import text_util

pos = text_util.load_txt("D:\project\9跨领域情感分析\英文\data/all/dvd+electronics.txt")
pos_list = text_util.seg_words(pos)

# 创建词向量模型 由于语料库样本少 保留全部词汇进行训练
model = Word2Vec(pos_list, sg=1, size=300, window=5, min_count=1, negative=3, sample=0.001, hs=1, workers=4)

# 检测词向量之间关系
print(model.similarity(u"good", u"nice"))

# 保存模型
model.save("D:\project\9跨领域情感分析\英文/2词向量/300/dvd+electronics.vec")
import gensim
# word2vec = gensim.models.KeyedVectors.load_word2vec_format('book+dvd.txt',binary=True)
model = Word2Vec.load("D:\project\9跨领域情感分析\英文/2词向量/300/dvd+electronics.vec")

import os
os.system("F:\Video\logo.mp3")

