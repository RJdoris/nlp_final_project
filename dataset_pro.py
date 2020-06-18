import json
import re
import pandas as pd
import numpy as np
import pickle
import os
import random


def label_id():
    entities=["subject","body","decorate","frequency","item","disease"]
    id2label=[]
    id2label.append("o")
    for entitie in entities:
        id2label.append("B-"+entitie)
        id2label.append("I-"+entitie)

    label2id={id2label[i]:i for i in range(len(id2label))}

    return label2id


def change(matched):
    content = matched.group("content")
    while (content[-1].isdigit()):
        content = content[:-1]

    return content


def process_data(path):
    model_data={}
    data_list=[]
    with open(path,'r',encoding="utf8") as f:
        json_data=json.load(f)

    for data in json_data:
        text=data["text"]
        pattern_first=r"\[(?P<content>\D+?)\][a-z]{3}"
        pattern_second=r"\[\d*(?P<content>21-三体[^\]]*|18-三体[^\]]*|18号[^\]]*|21-羟化酶[^\]]*)\][a-z]{3}"
        pattern_third=r"\[\d+(?P<content>[^\]]*)\][a-z]{3}"
        out=text.replace(" ","")
        out=out.replace(",", "，")
        out=re.sub(pattern_first,change,out)
        out=re.sub(pattern_second,change,out)
        out=re.sub(pattern_third,change,out)
        data_list.append(out)  #data_list中包含所有处理好的句子

    model_data["word"]=data_list
    model_data["label"]=prepare_tag(json_data,data_list)

    num_samples = len(data_list)
    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in model_data.values()]))  # 解压
        dataset += records+[['sep']]  # 每存完一个句子需要一行空行进行隔离
    dataset = dataset[:-1]
    dataset = pd.DataFrame(dataset)  # 转换成dataframe
    save_path = path.split(".")[0]
    save_path = save_path + ".csv"
    dataset.to_csv(save_path, index=False, encoding='utf-8')
    return save_path

def read_data(data_path):
    data = []
    with open(data_path, encoding='utf-8') as fr:
        lines = fr.readlines()
        #print(lines)
    sent_, tag_ = [], []
    for line in lines:
        line=line[:-1]
        if line != 'sep,':
            [char, label] = line.strip().split(",")
            sent_.append(char)
            tag_.append(label)
        else:
            data.append((sent_, tag_))
            sent_, tag_ = [], []

    return data  # [(句子，标签),(句子，标签)...]


def word_id(vocab_path, data_path, min_count):
    data = read_data(data_path)
    word2id = {}
    for sent_, tag_ in data:
        for word in sent_:
            if word.isdigit():
                word = '<NUM>'
            elif ('\u0041' <= word <='\u005a') or ('\u0061' <= word <='\u007a'):
                word = '<ENG>'
            if word not in word2id:
                word2id[word] = [len(word2id)+1, 1]  #word2id:{字:[id,个数]}
            else:
                word2id[word][1] += 1
    low_freq_words = []
    for word, [word_id, word_freq] in word2id.items():
        if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
            low_freq_words.append(word)
    for word in low_freq_words:
        del word2id[word]

    new_id = 1
    for word in word2id.keys():
        word2id[word] = new_id
        new_id += 1
    word2id['<UNK>'] = new_id
    word2id['<PAD>'] = 0

    print(len(word2id))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)


def sentence_id(sent, word2id):
    sentence_id = []
    for word in sent:
        if word.isdigit():
            word = '<NUM>'
        elif ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
            word = '<ENG>'
        if word not in word2id:
            word = '<UNK>'
        sentence_id.append(word2id[word])
    return sentence_id  #把一个句子转化成id序列


def read_dictionary(vocab_path):
    vocab_path = os.path.join(vocab_path)
    with open(vocab_path, 'rb') as fr:
        word2id = pickle.load(fr)
    print('vocab_size:', len(word2id))
    return word2id


def random_embedding(vocab, embedding_dim):
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def pad_sequences(sequences, pad_mark=0):
    max_len = max(map(lambda x : len(x), sequences))
    seq_list, seq_len_list = [], []
    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_len] + [pad_mark] * max(max_len - len(seq), 0)
        seq_list.append(seq_)
        seq_len_list.append(min(len(seq), max_len))  #记录句子的真实长度
    return seq_list, seq_len_list


def batch_yield(data, batch_size, word2id, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence_id(sent_, word2id)
        label_ = [tag2label[tag] for tag in tag_]

        if len(seqs) == batch_size:
            yield seqs, labels
            seqs, labels = [], []

        seqs.append(sent_)
        labels.append(label_)

    if len(seqs) != 0:
        yield seqs, labels

def data_generate(data, word2id, tag2label, shuffle=False):
    if shuffle:
        random.shuffle(data)

    seqs, labels = [], []
    for (sent_, tag_) in data:
        sent_ = sentence_id(sent_, word2id)
        label_ = [tag2label[tag] for tag in tag_]
        seqs.append(sent_)
        labels.append(label_)


    return seqs, labels


def prepare_tag(json_data,data_list):
    #提取所有的属性特征并按照pos值给句子打标签
    tags=[]
    i=0
    for data in json_data:
        keys=data["symptom"].keys()
        tag=["o" for s in data_list[i]]
        for key in keys:
            if(data["symptom"][key]["has_problem"]):
                continue
            else:
                pos=data["symptom"][key]["subject"]["pos"]
                if(len(pos)!=0):
                    j=0
                    while(j<len(pos)-1):
                        tag[pos[j]]="B-subject"
                        for p in range(pos[j]+1,pos[j+1]+1):
                            tag[p]="I-subject"
                        j+=2

                pos=data["symptom"][key]["body"]["pos"]
                if (len(pos) != 0):
                    j = 0
                    while (j < len(pos) - 1):
                        tag[pos[j]] = "B-body"
                        for p in range(pos[j] + 1, pos[j + 1] + 1):
                            tag[p] = "I-body"
                        j += 2

                pos=data["symptom"][key]["decorate"]["pos"]
                if (len(pos) != 0):
                    j = 0
                    while (j < len(pos) - 1):
                        tag[pos[j]] = "B-decorate"
                        for p in range(pos[j] + 1, pos[j + 1] + 1):
                            tag[p] = "I-decorate"
                        j += 2

                pos=data["symptom"][key]["frequency"]["pos"]
                if (len(pos) != 0):
                    j = 0
                    while (j < len(pos) - 1):
                        tag[pos[j]] = "B-frequency"
                        for p in range(pos[j] + 1, pos[j + 1] + 1):
                            tag[p] = "I-frequency"
                        j += 2

                pos=data["symptom"][key]["item"]["pos"]
                if (len(pos) != 0):
                    j = 0
                    while (j < len(pos) - 1):
                        tag[pos[j]] = "B-item"
                        for p in range(pos[j] + 1, pos[j + 1] + 1):
                            tag[p] = "I-item"
                        j += 2

                pos=data["symptom"][key]["disease"]["pos"]
                if (len(pos) != 0):
                    j = 0
                    while (j < len(pos) - 1):
                        tag[pos[j]] = "B-disease"
                        for p in range(pos[j] + 1, pos[j + 1] + 1):
                            tag[p] = "I-disease"
                        j += 2

        tags.append(tag)
        #print(tag)
        #print(i)
        i+=1
    return tags


if __name__ == '__main__':
    word2id=read_dictionary("train.pkl")
    print(word2id)
    # label2id=label_id()
    # print(label2id)

