# !/usr/bin/env python
# coding:utf-8
# Author: Qiao Ruijie

import json
import numpy as np
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import Model, Input
from keras.layers import Dense, Bidirectional, Dropout, LSTM, TimeDistributed, Masking
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from seqeval.metrics import classification_report
import matplotlib.pyplot as plt
import pickle

from dataset import data_trans
from dataset_pro import read_dictionary,random_embedding,label_id,read_data,data_generate


MAX_SEQ_LEN = 200 #训练集中最长语句长度为1080

# 读取测试集原始数据
label2id=label_id()
test_data=read_data("dataset_pro/test.csv")




# 读取训练集，验证集和测试集编码数据
with open('dataset/train_encode_text.txt', 'rb') as f:
    train_x = pickle.load(f)

with open('dataset/train_encode_tag.txt', 'rb') as f:
    train_y = pickle.load(f)

with open('dataset/dev_encode_text.txt', 'rb') as f:
    dev_x = pickle.load(f)

with open('dataset/dev_encode_tag.txt', 'rb') as f:
    dev_y = pickle.load(f)

with open('dataset/test_encode_text.txt', 'rb') as f:
    test_x = pickle.load(f)



# 读取label2id字典
"""label_id_dict = {
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
}"""

id_label_dict = {v:k for k,v in label2id.items()}



# Build model
def build_model(max_para_length, n_tags):
    # Bert Embeddings
    #bert_output = Input(shape=(max_para_length, 768, ), name="bert_output")
    bert_output = Input(shape=(max_para_length, 312,), name="bert_output")
    # LSTM model
    lstm = Bidirectional(LSTM(units=300, return_sequences=True), name="bi_lstm")(bert_output)
    drop = Dropout(0.5, name="dropout")(lstm)
    dense = TimeDistributed(Dense(n_tags, activation="softmax"), name="time_distributed")(drop)
    crf = CRF(n_tags)
    out = crf(dense)
    model = Model(inputs=bert_output, outputs=out)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    # 模型结构总结
    model.summary()
    #plot_model(model, to_file="albert_bi_lstm.png", show_shapes=True)

    return model


# 模型训练
def train_model():

    # 读取训练集，验证集和测试集数据

    # 模型训练
    model = build_model(MAX_SEQ_LEN, len(label2id.keys())+1)
    # early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
    history = model.fit(train_x, train_y, validation_data=(dev_x, dev_y), batch_size=32, epochs=60, callbacks=[early_stopping])

    model.save("models/bert_ner_e60.h5")

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
    plt.savefig("models/bert_loss_acc_e60.png")

    # 模型在测试集上的表现
    # 预测标签

    y = np.argmax(model.predict(test_x), axis=2)
    pred_tags = []
    for i in range(y.shape[0]):
        pred_tags.append([id_label_dict[_] for _ in y[i] if _])

    # 因为存在预测的标签长度与原来的标注长度不一致的情况，因此需要调整预测的标签
    #test_sents, test_tags = origin_test_X, origin_test_y
    test_sents = []
    test_tags = []
    for (sent_, tag_) in test_data:
        test_sents.append(sent_)
        test_tags.append(tag_)
    final_tags = []
    for test_tag, pred_tag in zip(test_tags, pred_tags):
        if len(test_tag) == len(pred_tag):
            final_tags.append(pred_tag)
        elif len(test_tag) < len(pred_tag):
            final_tags.append(pred_tag[:len(test_tag)])
        else:
            final_tags.append(pred_tag + ['o'] * (len(test_tag) - len(pred_tag)))

    # 利用seqeval对测试集进行验证
    print(classification_report(test_tags, final_tags, digits=4))


if __name__ == '__main__':
    train_model()