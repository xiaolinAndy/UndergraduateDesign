import pickle
import random
import re
import sys
import string
from collections import defaultdict
reload(sys)
sys.setdefaultencoding('utf-8')

state = 'tieba'
'''if state == 'twitter':
    file_name = 'D:\Andy\Code\data\\twitter_big_corpus_cleaned_2.pkl'
    file_name_1 = 'D:\Andy\Code\data\\twitter_big_corpus_cleaned_500K_train.pkl'
    file_name_2 = 'D:\Andy\Code\data\\twitter_big_corpus_cleaned_50K_valid.pkl'
    file_name_3 = 'D:\Andy\Code\data\\twitter_big_corpus_cleaned_50K_test.pkl'
    dialog = []
    flag = 0
    count1 = 0
    count2 = 0
    max_length = 0
    min_length = 10000
    total_length = 0
    words = {}

    f1 = open(file_name_1, 'wb')
    f2 = open(file_name_2, 'wb')
    f3 = open(file_name_3, 'wb')
    with open(file_name, 'rb') as f:
        D = pickle.load(f)
        for i, d in enumerate(D['data']):
            line1 = d[0]
            line2 = d[1]
            words1 = line1.split()
            words2 = line2.split()
            for w in words1:
                words[w] = words.get(w, 0) + 1
            for w in words2:
                words[w] = words.get(w, 0) + 1
            if (i % 10000 == 0):
                print(i)
        #pickle.dump(D, ff, protocol=2)

        for i, d in enumerate(D['data']):
            line1 = d[0]
            line2 = d[1]
            words1 = line1.split()
            words2 = line2.split()
            for w in words1:
                if words[w] == 1:
                    flag = 1
                    break
            for w in words2:
                if words[w] == 1:
                    flag = 1
                    break
            if flag == 0:
                dialog.append(d)
            flag = 0
            if (i % 10000 == 0):
                print(i)

        dialog1 = random.sample(dialog, 600000)
        data = {'name': 'twitter_big_corpus_cleaned_500K_train',
                'count': 500000,
                'data': dialog1[:500000]}
        print(data['data'][0])
        pickle.dump(data, f1, protocol=2)
        data = {'name': 'twitter_big_corpus_cleaned_50K_valid',
                'count': 50000,
                'data': dialog1[500000:550000]}
        pickle.dump(data, f2, protocol=2)
        print(data['data'][0])
        data = {'name': 'twitter_big_corpus_cleaned_50K_test',
                'count': 50000,
                'data': dialog1[550000:]}
        pickle.dump(data, f3, protocol=2)
else:
    file_name = 'D:\Andy\Data\\tieba\comments_final_modified.pkl'
    file_name_1 = 'D:\Andy\Code\data\\tieba_500K_train.pkl'
    file_name_2 = 'D:\Andy\Code\data\\tieba_50K_valid.pkl'
    file_name_3 = 'D:\Andy\Code\data\\tieba_50K_test.pkl'
    dialog = []
    flag = 0
    count1 = 0
    count2 = 0
    max_length = 0
    min_length = 10000
    total_length = 0
    words = {}

    with open(file_name, 'rb') as f:
        D = pickle.load(f)
        for i, d in enumerate(D):
            line1 = d[0]
            line2 = d[1]
            for w in line1:
                words[w] = words.get(w, 0) + 1
            for w in line2:
                words[w] = words.get(w, 0) + 1
            if (i % 10000 == 0):
                print(i)
        # pickle.dump(D, ff, protocol=2)

        for i, d in enumerate(D):
            line1 = d[0]
            line2 = d[1]
            for w in line1:
                if words[w] == 1:
                    flag = 1
                    break
            for w in line2:
                if words[w] == 1:
                    flag = 1
                    break
            if flag == 0:
                max_length = max(max_length, len(d[0]), len(d[1]))
                min_length = min(min_length, len(d[0]), len(d[1]))
                dialog.append(d)
            flag = 0
            if (i % 10000 == 0):
                print(i)
        print(max_length, min_length)
        
        dialog1 = random.sample(dialog, 550000)
        data = {'name': 'tieba_500K_train',
                'count': 500000,
                'data': dialog1[:500000]}
        print(data['data'][0])
        pickle.dump(data, f1, protocol=2)
        data = {'name': 'tieba_50K_valid',
                'count': 40000,
                'data': dialog1[500000:540000]}
        pickle.dump(data, f2, protocol=2)
        print(data['data'][0])
        data = {'name': 'tieba_50K_test',
                'count': 10000,
                'data': dialog1[540000:550000]}
        pickle.dump(data, f3, protocol=2)
        print(data['data'][0])


    print(len(words))
    for word in words:
        count2 += words[word]
        if (words[word] >= 20):
            count1 += 1
    print(count1, count2)'''
#utterance = utterance.replace('  ', ' ')

'''ind = pickle.load(open('./data/index', 'rb'))
f = open('./data/index2.pkl', 'wb')
pickle.dump(ind, f, protocol=2)

file_name = 'D:\Andy\Code\data\\tieba_500K_train.pkl'
output = open('./data/triples.txt', 'w')
D = pickle.load(open(file_name, 'rb'))['data']
contexts = [d[0] for d in D]
responses = [d[1] for d in D]
for i, d in enumerate(D):
    if i % 10 == 0:
        print i
    try:
        index = (contexts[:i] + contexts[i+1:]).index(d[0])
        if index < i:
            r_p = responses[index]
        else:
            r_p = responses[index + 1]
        if r_p == d[1]:
            continue
        str = ' '.join(d[0]) + '\t' + ' '.join(d[1]) + '\t' + ' '.join(r_p)
        str = re.sub('<end>', '</s>', str)
        output.write(str + '\n')
    except:
        continue'''

'''embs = {}
embs['c'] = open('./tieba/clean.tieba.contexts.txt', 'r').readlines()
embs['r_gt'] = open('./tieba/clean.true.responses.txt', 'r').readlines()
embs['r_models'] = {}
embs['r_models']['tfidf'] = open('./tieba/clean.tfidf.responses.txt', 'r').readlines()
embs['r_models']['de'] = open('./tieba/clean.dual_encoder.responses.txt', 'r').readlines()
embs['r_models']['vhred'] = open('./tieba/clean.VHRED.responses.txt', 'r').readlines()
embs['r_models']['human'] = open('./tieba/clean.human.responses.txt', 'r').readlines()
f = open('./tieba/clean.total.txt', 'w')
r_models = embs['r_models']['tfidf'] + embs['r_models']['de'] + embs['r_models']['vhred'] + embs['r_models']['human']

for i in range(len(r_models)):
    str = embs['c'][i % len(embs['c'])].strip() + '\t' + embs['r_gt'][i % len(embs['c'])].strip() + '\t' + r_models[i].strip() + '\n'
    f.write(str)'''







