import pickle
import re
import sys
import string

file_name = 'D:\Andy\Code\data\\twitter_big_corpus_cleaned_2.pkl'
dialog = []
count = 0
max_length = 0
min_length = 10000
total_length = 0
words = {}

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

print(len(words))
for word in words:
    if (words[word] == 1):
        count += 1
print(count)