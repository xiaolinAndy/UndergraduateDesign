import pickle
import random
import re
import sys
import string
from collections import defaultdict

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

'''file_name = '../Data/tieba/comments_final_modified.pkl'
tieba_context = open('./tieba/new.tieba.contexts.txt', 'r', encoding='utf-8').readlines()
tieba_response = open('./tieba/new.true.responses.txt', 'r', encoding='utf-8').readlines()
output = open('./tieba/new.human.responses.txt', 'w')
f = pickle.load(open(file_name, 'rb'))
count = 0
context = []
response = []
for d in f:
    context.append(''.join(d[0]))
    response.append(''.join(d[1]))
#print(context[0])
d = defaultdict(list)
for k,va in [(v,i) for i,v in enumerate(context)]:
    d[k].append(va)
for i, line in enumerate(tieba_context):
    flag = 0
    #print(line)
    line = line.replace(' ', '').strip()
    index = d[line]
    tr = tieba_response[i].replace(' ', '').strip()
    if len(index) > 1:
        for j in index:
            if tr != response[j] and response[j].find('撸') == -1 and response[j].find('十五字') == -1:
                output.write(response[j] + '\n')
                flag = 1
                count += 1
                break
    if not flag:
        print(line, '\n')
        res = input("请输入回答: ")
        output.write(res + '\n')
print(count)'''
ind = pickle.load(open('./data/index', 'rb'))
f = open('./data/index2.pkl', 'wb')
pickle.dump(ind, f, protocol=2)




