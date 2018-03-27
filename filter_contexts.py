import numpy as np
import re

'''eliminate_index = []
tieba_context = open('./tieba/new.tieba.contexts.txt', 'r', encoding='utf-8').readlines()
for i, line in enumerate(tieba_context):
    line = re.sub(" ", '', line)
    line = re.sub("<end>", '', line)
    print('(%d/%d) A: %s' % (i + 1, 500, line))
    res = input("请评分: ")
    if res != '':
        eliminate_index.append(i)
        print(i, len(eliminate_index))

np.save('./tieba/eliminate.index.txt', eliminate_index)'''
index = np.load('./tieba/eliminate.index.txt.npy')
index = list(index)
index.append(233)
index.append(313)
index.append(359)
tieba_context = open('./tieba/new.human.responses.txt', 'r').readlines()
final_context = open('./tieba/final.human.responses.txt', 'w')
for i,line in enumerate(tieba_context):
    if i not in index:
        final_context.write(line)