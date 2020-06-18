# !/usr/bin/env python
# coding:utf-8
# Author: Qiao Ruijie

import numpy as np
from keras.utils import to_categorical
import pickle
from tqdm import tqdm
#from bert.extract_feature import BertVector
from albert_zh.extract_feature import BertVector
from dataset_pro import read_dictionary,random_embedding,label_id,read_data,data_generate
from dataset import data_trans


MAX_SEQ_LEN = 200 #训练集中最长语句长度为1080


# 读取训练集，验证集和测试集原始数据

label2id=label_id()
train_data=read_data("dataset_pro/train.csv")
dev_data=read_data("dataset_pro/dev.csv")
test_data=read_data("dataset_pro/test.csv")
word2id=read_dictionary("dataset_pro/train.pkl")


_, origin_train_X, origin_train_y = data_trans('dataset/train.txt')
_, origin_dev_X, origin_dev_y = data_trans('dataset/dev.txt')
_, origin_test_X, origin_test_y = data_trans('dataset/test.txt')


train_sent = []
train_tag = []
for (sent_, tag_) in train_data:
    train_sent.append(''.join(sent_))
    train_tag.append(tag_)

dev_sent = []
dev_tag = []
for (sent_, tag_) in dev_data:
    dev_sent.append(''.join(sent_))
    dev_tag.append(tag_)

test_sent = []
test_tag = []
for (sent_, tag_) in test_data:
    test_sent.append(''.join(sent_))
    test_tag.append(tag_)

#train_X, train_Y = data_generate(train_data,word2id,label2id)
#dev_X, dev_Y = data_generate(dev_data,word2id,label2id)
#test_X, test_Y = data_generate(test_data,word2id,label2id)

# 利用ALBERT提取文本特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
f = lambda text: bert_model.encode([text])["encodes"][0]



# 载入数据
def input_data(sentences, tags):

    #sentences, tags = read_data(file_path)
    #print("sentences length: %s " % len(sentences))
    #print("last sentence: ", sentences[-1])

    # BERT ERCODING
    print("start BERT encding")
    x = []
    processor_bar = tqdm(sentences)
    for bar, sent in zip(processor_bar, sentences):
        processor_bar.set_description("Processing")
        x.append(f(sent))

    x = np.array(x)
    print("end BERT encoding")

    # 对y值统一长度为MAX_SEQ_LEN
    new_y = []
    for seq in tags:
        num_tag = [label2id[_] for _ in seq]
        #num_tag = seq
        if len(seq) < MAX_SEQ_LEN:
            num_tag = num_tag + [0] * (MAX_SEQ_LEN-len(seq))
        else:
            num_tag = num_tag[: MAX_SEQ_LEN]

        new_y.append(num_tag)

    # 将y中的元素编码成ont-hot encoding
    y = np.empty(shape=(len(tags), MAX_SEQ_LEN, len(label2id.keys())+1))

    for i, seq in enumerate(new_y):
        y[i, :, :] = to_categorical(seq, num_classes=len(label2id.keys())+1)

    return x, y


train_x, train_y = input_data(train_sent, train_tag)
dev_x, dev_y = input_data(dev_sent, dev_tag)
test_x, test_y = input_data(test_sent, test_tag)

with open('dataset/train_encode_text.txt', 'wb') as f:
    pickle.dump(train_x, f)

with open('dataset/train_encode_tag.txt', 'wb') as f:
    pickle.dump(train_y, f)

with open('dataset/dev_encode_text.txt', 'wb') as f:
    pickle.dump(dev_x, f)

with open('dataset/dev_encode_tag.txt', 'wb') as f:
    pickle.dump(dev_y, f)

with open('dataset/test_encode_text.txt', 'wb') as f:
    pickle.dump(test_x, f)

with open('dataset/test_encode_tag.txt', 'wb') as f:
    pickle.dump(test_y, f)