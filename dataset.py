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

    return new_list, text_list, tag_list





def spo_generate(new_list, text_list):
    data_list = []

    for idx, value in enumerate(new_list):
        data = {}
        spo_list = []
        data['text'] = text_list[idx]
        sym_list = value['symptom']
        for sym in sym_list.keys():
            if sym_list[sym]['has_problem'] == True:
                break
            subject = sym_list[sym]['self']['val']
            SUB_VAL = sym_list[sym]['subject']['val']
            BOD_VAL = sym_list[sym]['body']['val']
            DEC_VAL = sym_list[sym]['decorate']['val']
            FRE_VAL = sym_list[sym]['frequency']['val']
            ITE_VAL = sym_list[sym]['item']['val']
            DIS_VAL = sym_list[sym]['disease']['val']

            if SUB_VAL != '':
                spo_list.append([subject, 'SUB', SUB_VAL])
            if BOD_VAL != '':
                spo_list.append([subject, 'BOD', BOD_VAL])
            if DEC_VAL != '':
                spo_list.append([subject, 'DEC', DEC_VAL])
            if FRE_VAL != '':
                spo_list.append([subject, 'FRE', FRE_VAL])
            if ITE_VAL != '':
                spo_list.append([subject, 'ITE', ITE_VAL])
            if DIS_VAL != '':
                spo_list.append([subject, 'DIS', DIS_VAL])

        data['spo_list'] = spo_list
        data_list.append(data)

    return data_list

