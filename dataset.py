# !/usr/bin/env python
# coding:utf-8
# Author: Qiao Ruijie
import json
import re


def data_trans(input_file):
    with open(input_file, 'r') as f:
        content = f.read()

    #fw = open(output_file, 'w')

    a = content.split('text')

    origin_list = []

    for i in range(1, len(a) - 1):
        a[i] = a[i].rstrip(',\n     {\n          "')
        a[i] = a[i].lstrip('"')
        item = '{"text"' + a[i]
        origin_list.append(json.loads(item))

    a[-1] = a[-1].rstrip('\n]')
    a[-1] = '{"text' + a[-1]

    origin_list.append(json.loads(a[-1]))
    new_list = []
    text_list = []




    for origin_data in origin_list:
        text = origin_data['text']
        if re.search(r'\[\d{4,}', text) or re.search(r'\d{4,}\]', text):      #去掉【】内部连着数字的情况
            #origin_list.pop(index)
            continue
        else:
            text = re.sub(r'(\d+(?=\]))|((?<=\[)\d{,3})|(dis)|(sym)|(bod)|(ite)|[\[\]]', '', text)
            text = re.sub(' ', '', text)
            text_list.append(text)
            #fw.write(text)
            #fw.write('\n')
            new_list.append(origin_data)


    features = ['self','subject','body','decorate','frequency','item','disease']
    fea_abbr = {'subject': 'SUB', 'body': 'BOD', 'decorate': 'DEC', 'frequency': 'FRE', 'item': 'ITE', 'disease': 'DIS'}



    tag_list = []
    for index, data in enumerate(new_list):
        sym_list = data['symptom']
        sent = text_list[index]
        tag = ['O' for index in range(len(sent))]

        for key in sym_list.keys():
            if sym_list[key]['has_problem'] == True:
                continue
            else:
                for i in features:
                    pos = sym_list[key][i]['pos']
                    val = sym_list[key][i]['val'].split()
                    for n in range(0, int(len(pos) / 2)):
                        try:
                            sub_val = val[n]
                            sub_pos = pos[n * 2:n * 2 + 2]
                        except IndexError:
                            # print("数组越界")
                            new_list.pop(index)
                            text_list.pop(index)
                        else:
                            if (sub_pos[1] - sub_pos[0] + 1) != len(sub_val):
                                # print('error1')
                                continue
                            elif sent[sub_pos[0]: sub_pos[1] + 1] != sub_val:
                                # print('error2')
                                continue
                            elif i != 'self':
                                tag[sub_pos[0]] = 'B-' + fea_abbr[i]
                                for t in range(sub_pos[0] + 1, sub_pos[1] + 1):
                                    tag[t] = 'I-' + fea_abbr[i]
        tag_list.append(tag)

    #print(len(text_list))
    #print(len(tag_list))

    assert len(text_list) == len(tag_list)

    return text_list, tag_list





def create_dico(item_list):
    """
    对于item_list中的每一个items，统计items中item在item_list中的次数
    item:出现的次数
    :param item_list:
    :return:
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico: #第一次出现，标记为1
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    创建item to id, id_to_item
    item的排序按词典中出现的次数
    :param dico:
    :return:
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def word_mapping(sentences):
    """
    构建字典
    :param sentences:
    :return:
    """
    word_list = [[x for x in s] for s in sentences]  # 得到所有的字
    dico = create_dico(word_list)
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = create_mapping(dico)
    return dico, word_to_id, id_to_word


def tag_mapping(tag_sentences):
    """
    构建标签字典
    :param sentences:
    :return:
    """
    tag_list = [[x for x in s] for s in tag_sentences]
    dico = create_dico(tag_list)
    tag_to_id, id_to_tag = create_mapping(dico)
    return dico, tag_to_id, id_to_tag


sents, tags = data_trans('dataset/test.txt')

print(sents[0])
print(tags[0])

#with open('dataset/train_text.txt', 'r') as f:
#    content = f.read()
#sentences = content.split('\n')
#sentences.pop(-1)
#print(len(sentences))  #4145
#print(len(data_list))

#dico, word_to_id, id_to_word = word_mapping(sentences)
#print(word_to_id)

#data_tag_trans(data_list, sentences, 'train_tag.txt')

