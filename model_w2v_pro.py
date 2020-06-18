# !/usr/bin/env python
# coding:utf-8
# Author: Qiao Ruijie

import keras
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import Model, Input, Sequential
from keras.layers import Embedding, Dense, Bidirectional, Dropout, LSTM, TimeDistributed, Masking
from keras.utils import to_categorical, plot_model
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt




from dataset_pro import read_dictionary,random_embedding,label_id,read_data,data_generate


MAX_SEQ_LEN = 200 #训练集中最长语句长度为1080


# 读取训练集，验证集和测试集原始数据

label2id=label_id()
train_data=read_data("dataset_pro/train.csv")
dev_data=read_data("dataset_pro/dev.csv")
test_data=read_data("dataset_pro/test.csv")
word2id=read_dictionary("dataset_pro/train.pkl")

train_X, train_Y = data_generate(train_data,word2id,label2id)
dev_X, dev_Y = data_generate(dev_data,word2id,label2id)
test_X, test_Y = data_generate(test_data,word2id,label2id)


# 利用ALBERT提取文本特征

# label2id字典
label_id_dict = {
  "O": 1,
  "B-SUB": 2,
  "I-SUB": 3,
  "B-BOD": 4,
  "I-BOD": 5,
  "B-DEC": 6,
  "I-DEC": 7,
  "B-FRE": 8,
  "I-FRE": 9,
  "B-ITE": 10,
  "I-ITE": 11,
  "B-DIS": 12,
  "I-DIS": 13,
}

id_label_dict = {v:k for k,v in label_id_dict.items()}

#词向量训练,返回序列化的输入值和embedding矩阵
def w2v ():

    sentences = []
    for (sent_, tag_) in train_data:
        sentences.append(sent_)

    train = pad_sequences(train_X, maxlen=MAX_SEQ_LEN, padding='post')  #序列化的语料
    dev = pad_sequences(dev_X, maxlen=MAX_SEQ_LEN, padding='post')
    test = pad_sequences(test_X, maxlen=MAX_SEQ_LEN, padding='post')

    model = Word2Vec(sentences, size=100, min_count=5)
    # word_vec_dict = dict((k, model.wv[k]) for k, v in model.wv.vocab.items())
    # model.save( 'word2vec')

    embedding_word2vec_matrix = np.zeros((len(word2id) + 1, 100))
    for word, i in word2id.items():
        embedding_vector = model[word] if word in model else None
        if embedding_vector is not None:
            embedding_word2vec_matrix[i] = embedding_vector
        else:
            unk_vec = np.random.random(100) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_word2vec_matrix[i] = unk_vec

    return train, dev, test, embedding_word2vec_matrix

train,dev,test,matrix = w2v()

# 将标签序列y转化为one-hot编码
def one_hot_trans(tags):

    # 对y值统一长度为MAX_SEQ_LEN
    new_y = []
    for seq in tags:
        num_tag = seq
        if len(seq) < MAX_SEQ_LEN:
            num_tag = num_tag + [0] * (MAX_SEQ_LEN-len(seq))
        else:
            num_tag = num_tag[: MAX_SEQ_LEN]

        new_y.append(num_tag)

    np.array(new_y)

    # 将y中的元素编码成ont-hot encoding
    y = np.empty(shape=(len(tags), MAX_SEQ_LEN, len(label_id_dict.keys())+1))


    for i, seq in enumerate(new_y):
        y[i, :, :] = to_categorical(seq, num_classes=len(label_id_dict.keys())+1)

    return y


# Build model
def build_model(max_para_length, n_tags, embedding_matrix):

    input_layer = Input(shape=(MAX_SEQ_LEN,))
    embedding_layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        input_length=MAX_SEQ_LEN,
        weights=[embedding_matrix],
        trainable=False)   #trainable=False使用预训练的词向量
    bi_lstm_layer = Bidirectional(LSTM(300, return_sequences=True))
    bi_lstm_drop_layer = Dropout(0.5)
    dense_layer = TimeDistributed(Dense(n_tags, activation="softmax"))
    crf_layer = CRF(n_tags)

    input = input_layer
    embedding = embedding_layer(input)
    bi_lstm = bi_lstm_layer(embedding)
    bi_lstm_drop = bi_lstm_drop_layer(bi_lstm)
    dense = dense_layer(bi_lstm_drop)
    crf = crf_layer(dense)

    model = Model(input=[input], output=[crf])
    optmr = keras.optimizers.Adam(lr=0.001, beta_1=0.5)
    model.compile(optimizer=optmr, loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])




    # 模型结构总结
    model.summary()
    #plot_model(model, to_file="lstm_crf.png", show_shapes=True)

    return model


# 模型训练
def train_model():

    # 得到模型可直接使用的训练集，验证集和测试集数据
    train_x, dev_x, test_x, embedding_matrix = w2v()
    train_y = one_hot_trans(train_Y)
    dev_y = one_hot_trans(dev_Y)
    #test_y = one_hot_trans(origin_test_y)



    # 模型训练
    model = build_model(MAX_SEQ_LEN, len(label_id_dict.keys())+1, embedding_matrix)
    history = model.fit(train_x, train_y, validation_data=(dev_x, dev_y), batch_size=32, epochs=60)

    model.save("lstm_crf_w2v_ner_0618.h5")

    # 绘制loss和acc图像
    plt.subplot(2, 1, 1)
    epochs = len(history.history['loss'])
    plt.plot(range(epochs), history.history['loss'], label='loss')
    plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    epochs = len(history.history['crf_viterbi_accuracy'])
    plt.plot(range(epochs), history.history['crf_viterbi_accuracy'], label='crf_viterbi_accuracy')
    plt.plot(range(epochs), history.history['val_crf_viterbi_accuracy'], label='val_crf_viterbi_accuracy')
    plt.legend()
    plt.savefig("lstm_crf_w2v_loss_acc_0618.png")

    # 模型在测试集上的表现
   
    # 预测标签
    """
    y = np.argmax(model.predict(test_x), axis=2)
    pred_tags = []
    for i in range(y.shape[0]):
        pred_tags.append([id_label_dict[_] for _ in y[i] if _])

    # 因为存在预测的标签长度与原来的标注长度不一致的情况，因此需要调整预测的标签
    test_sents, test_tags = origin_test_X, origin_test_y
    final_tags = []
    for test_tag, pred_tag in zip(test_tags, pred_tags):
        if len(test_tag) == len(pred_tag):
            final_tags.append(pred_tag)
        elif len(test_tag) < len(pred_tag):
            final_tags.append(pred_tag[:len(test_tag)])
        else:
            final_tags.append(pred_tag + ['O'] * (len(test_tag) - len(pred_tag)))

    # 利用seqeval对测试集进行验证
    print(classification_report(test_tags, final_tags, digits=4))
    """

if __name__ == '__main__':
    train_model()



