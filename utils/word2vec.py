from gensim.models.word2vec import Word2Vec
import text_util

pos = text_util.load_txt("../raw_data/all/dvd+electronics.txt")
pos_list = text_util.seg_words(pos)

# 创建词向量模型 由于语料库样本少 保留全部词汇进行训练
model = Word2Vec(pos_list, sg=1, size=300, window=5, min_count=1, negative=3, sample=0.001, hs=1, workers=4)

# 检测词向量之间关系
print(model.similarity(u"good", u"nice"))

# 保存模型
model.save("..//dvd+electronics.vec")
model = Word2Vec.load("../w2v/dvd+electronics.vec")
