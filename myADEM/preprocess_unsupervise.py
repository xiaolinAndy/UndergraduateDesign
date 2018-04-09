import jieba
import re
import cPickle
import random
import sys
from experiments import default_config
import numpy as np
reload(sys)
sys.setdefaultencoding('utf-8')

def clean_data(text):
    clean_text = []
    for i in range(len(text)):
        s = re.sub('<end>', '', text[i].strip())
        s = re.sub(' ', '', s)
        if not s[-5:] == '<end>':
            s = s + '<end>'
        cut = jieba.cut(s)
        l = []
        new = []
        index = 0
        for w in cut:
            l.append(w)
        while index < len(l):
            if index + 2 < len(l) and l[index] == '<' and l[index + 1] == 'num' and l[index + 2] == '>':
                new.append('<num>')
                index += 3
            elif index + 2 < len(l) and l[index] == '<' and l[index + 1] == 'end' and l[index + 2] == '>':
                new.append('<end>')
                index += 3
            elif l[index] != ' ':
                new.append(l[index])
                index += 1
            else:
                index += 1
        clean_text.append(new)
    return clean_text

def clean_data_for_new_embedding(text):
    clean_text = []
    for i in range(len(text)):
        s = re.sub('<end>', '', text[i].strip())
        s = re.sub(' ', '', s)
        if not s[-5:] == '<end>':
            s = s + '<end>'
        cut = jieba.cut(s)
        l = []
        new = []
        index = 0
        for w in cut:
            l.append(w)
        while index < len(l):
            if index + 2 < len(l) and l[index] == '<' and l[index + 1] == 'num' and l[index + 2] == '>':
                new.append('<num>')
                index += 3
            elif index + 2 < len(l) and l[index] == '<' and l[index + 1] == 'end' and l[index + 2] == '>':
                new.append('</s>')
                index += 3
            elif l[index] != ' ':
                new.append(l[index])
                index += 1
            else:
                index += 1
        clean_text.append(new)
    return clean_text

def get_score(config):
    ##
    scores_avg = cPickle.load(open(config['score'], 'rb'))['avg']
    scores_1 = cPickle.load(open(config['score'], 'rb'))['score_1']
    scores_2 = cPickle.load(open(config['score'], 'rb'))['score_2']
    ind = cPickle.load(open(config['index'], 'rb'))['response_order']
    ordered_score, ordered_score_1, ordered_score_2 = range(len(ind)), range(len(ind)), range(len(ind))
    for i,j in enumerate(ind):
        ordered_score[j] = scores_avg[i]
        ordered_score_1[j] = scores_1[i]
        ordered_score_2[j] = scores_2[i]
    return ordered_score, ordered_score_1, ordered_score_2

def load_data(config):
    output = open('../tieba/evaluate.txt', 'w')
    contexts = open(config['contexts'], 'r').readlines()
    contexts = clean_data(contexts)
    true_res = open(config['true_responses'], 'r').readlines()
    true_res = clean_data(true_res)
    model_res = {}
    model_type = ['tfidf', 'de', 'vhred', 'human']
    for type in model_type:
        m_res = open(config[type], 'r').readlines()
        m_res = clean_data(m_res)
        model_res[type] = m_res
    score, score_1, score_2 = get_score(config)
    r_models = []
    for c, r, s in zip(contexts * 4, model_res['tfidf'] + model_res['de'] + model_res['vhred'] + model_res['human'], score):
        strs = ' '.join(c) + '\t' + ' '.join(r) + '\t' + str(s)
        strs = re.sub('<end>', '</s>', strs)
        output.write(strs + '\n')

def load_test_data(config):
    contexts = open('./test.txt', 'r').readlines()
    f = open('./cleaned_test.txt', 'w')
    contexts = clean_data_for_new_embedding(contexts)
    for c in contexts:
        f.write(c)
    #return contexts
if __name__ == "__main__":
    config = default_config()
    load_data(config)