import numpy as np
from collections import Counter
import random
import torch.nn.functional
import math
def load_data(data_path):
    data =[]
    ids =[]
    max_len =0
    law_labels =[]
    money_label =[]
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            lines = line.split('\t')
            ids.append(lines[0])

            sentence = lines[1].split(' ')
            tmp_len = len(sentence)
            if(tmp_len>max_len):
                max_len = tmp_len
            multi_label =[int(i)-1 for i in lines[3].split(',')]
            data.append(sentence)
            law_labels.append(multi_label)
            money_label.append(lines[2])
    return ids , data, money_label,law_labels

#对数据进行过采样
def over_sample(ids,data,labels):
    freqs = Counter(labels)
    max_count = freqs.most_common(1)[0][1]
    p ={ key: ((max_count-freqs[key])/freqs[key]) for key in freqs.keys() }
    new_data =[]
    new_labels =[]
    new_id =[]
    for i in range(len(labels)):
        cur_p =  p[labels[i]]
        if cur_p>1:
            for c in range(int(math.sqrt(cur_p)+0.5)):
                new_data.append(data[i])
                new_labels.append(labels[i])
                new_id.append(ids[i])
        else:
            rand_p = random.random()
            if rand_p >cur_p-0.1:
                new_data.append(data[i])
                new_labels.append(labels[i])
                new_id.append(ids[i])

    indices = np.arange(len(new_labels))
    np.random.shuffle(indices)
    return [new_id[i] for i in indices],[new_data[i] for i in indices],[new_labels[i] for i in indices]
def load_test_data(data_path):
    """
    载入测试数据
    """
    data= []
    tests_id = []
    max_sentence_len = 0
    with open(data_path, 'r') as f:
        for line in f:
            line_list = line.split('\t')
            tests_id.append(line_list[0])
            one_data = line_list[1].split(' ')
            tmp_len = len(one_data)
            if tmp_len > max_sentence_len:
                max_sentence_len = tmp_len
            data.append(one_data)
        f.close()
    print("max text length in test set: ", max_sentence_len)
    return tests_id, data

def bulid_vocabulary(data,min_count =3):
    count = [('<UNK>', -1), ('<PAD>', -1)]
    words =[]
    [words.extend(line) for line in data]
    counter = Counter(words)
    counter_list = counter.most_common()
    for word,c in counter_list:
        #记录最少出现三次的词
        if c>min_count:
            count.append((word,c))
        # 同理也可以限制最多出现的词的数目
    dict_word2index = {word:c for word,c in enumerate(count)}
    dict_index2word ={c:word for word,c in enumerate(count)}
    print("vocab size:", len(count))
    print(count[-1])
    return count, dict_word2index, dict_index2word

def bulid_dataset(ids,data,labels,dict_word2index,text_maxlen,num_classes):
    datasets=[]
    new_ids =[]
    new_labels =[]
    for i in range(len(labels)):
        new_ids.append('train_'+ids[i])
        multi_label = np.zeros(num_classes)
        for li in labels[i]:
            if(li>num_classes):
                print(li)
                labels[i].remove(li)
        multi_label[labels[i]] =1
        new_labels.append(multi_label)
        new_line =[]
        for word in data[i]:
            if word in dict_word2index:
                new_line.append(dict_word2index[word])
            else:
                new_line.append(0)#UNK
        pad_num = text_maxlen-len(new_line)
        while pad_num>0:
            new_line.append(1)
            pad_num-=1
        datasets.append(new_line[:max_text_len])
    return new_ids,np.array([datasets],dtype=np.int64),np.array(new_labels, dtype=np.int64)






