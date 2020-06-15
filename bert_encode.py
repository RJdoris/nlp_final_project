# !/usr/bin/env python
# coding:utf-8
# Author: Qiao Ruijie

import numpy as np
from keras.utils import to_categorical
import pickle

from bert.extract_feature import BertVector
from dataset import data_trans



MAX_SEQ_LEN = 200 #训练集中最长语句长度为1080

# 读取训练集，验证集和测试集原始数据
_, origin_train_X, origin_train_y = data_trans('dataset/train.txt')
_, origin_dev_X, origin_dev_y = data_trans('dataset/dev.txt')
_, origin_test_X, origin_test_y = data_trans('dataset/test.txt')

from tqdm import tqdm

# 利用ALBERT提取文本特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
f = lambda text: bert_model.encode([text])["encodes"][0]

# 读取label2id字典
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
        num_tag = [label_id_dict[_] for _ in seq]
        if len(seq) < MAX_SEQ_LEN:
            num_tag = num_tag + [0] * (MAX_SEQ_LEN-len(seq))
        else:
            num_tag = num_tag[: MAX_SEQ_LEN]

        new_y.append(num_tag)

    # 将y中的元素编码成ont-hot encoding
    y = np.empty(shape=(len(tags), MAX_SEQ_LEN, len(label_id_dict.keys())+1))

    for i, seq in enumerate(new_y):
        y[i, :, :] = to_categorical(seq, num_classes=len(label_id_dict.keys())+1)

    return x, y


train_x, train_y = input_data(origin_train_X, origin_train_y)
dev_x, dev_y = input_data(origin_dev_X, origin_dev_y)
test_x, test_y = input_data(origin_test_X, origin_test_y)

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