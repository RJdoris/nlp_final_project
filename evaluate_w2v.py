# !/usr/bin/env python
# coding:utf-8
# Author: Qiao Ruijie
import json
import numpy as np
from keras.models import load_model
from keras_contrib.layers.crf import CRF, crf_loss, crf_viterbi_accuracy
from seqeval.metrics import classification_report

from dataset import data_trans
from model import w2v
from model import label_id_dict


model = load_model('lstm_crf_ner_0610_3.h5',
                   custom_objects = {"CRF": CRF,
                                     'crf_loss': crf_loss,
                                     'crf_viterbi_accuracy': crf_viterbi_accuracy})


_, _, test_x, _ = w2v()
id_label_dict = {v:k for k,v in label_id_dict.items()}

y = np.argmax(model.predict(test_x), axis=2)
pred_tags = []
for i in range(y.shape[0]):
    pred_tags.append([id_label_dict[_] for _ in y[i] if _])

# 因为存在预测的标签长度与原来的标注长度不一致的情况，因此需要调整预测的标签
test_sents, test_tags = data_trans('dataset/test.txt')
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