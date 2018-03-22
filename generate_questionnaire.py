import pickle
import re

evaluate_num = 100
index = 1
task = pickle.load(open('./data/task_' + index, 'rb'))
output = open('./data/task2_tieba_%d.txt' % int(index) , 'a+')
for i in range(evaluate_num):
    c = re.sub('<end>', '', task['tieba'][i][0])
    r = re.sub('<end>', '', task['tieba'][i][1])
    c = re.sub(' ', '', c)
    r = re.sub(' ', '', r)
    output.write(str(i + 1) + ': ' + c + '\n')
    output.write(r + '\n')
