import jieba
import re
import cPickle
import numpy as np

def clean_data(text):
    clean_text = []
    for i in range(len(text)):
        s = re.sub('<end>', '', text[i])
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

def get_score(config):
    scores = cPickle.load(config['score'])['avg']
    ind = cPickle.load(config['index'])
    ordered_score = range(len(ind))
    for i,j in enumerate(ind):
        ordered_score[j] = scores[i]
    return ordered_score

def load_data(config):
    contexts = config['contexts'].readlines()
    contexts = clean_data(contexts)
    true_res = config['true_responses'].readlines()
    true_res = clean_data(true_res)
    model_res = {}
    model_type = ['tfidf', 'de', 'vhred', 'human']
    for type in model_type:
        m_res = config[type].readlines()
        m_res = clean_data(m_res)
        model_res[type] = m_res
    score = get_score(config)
    r_models = []
    for tr, dr, vr, hr, ts, ds, vs, hs in zip(model_res['tfidf'], model_res['de'], model_res['vhred'], model_res['human'], score[:387], score[387:387*2], score[387*2:387*3], score[387*3:]):
        r_models.append({'tfidf': (tr, ts),
                         'de': (dr, ds),
                         'vhred': (vr, vs),
                         'human': (hr, hs)})
    dataset = []
    for c, r_gt, r_m in zip(contexts, true_res, r_models):
        entry = {'c': c, 'r_gt': r_gt, 'r_models': r_m}
        dataset.append(entry)
    return dataset