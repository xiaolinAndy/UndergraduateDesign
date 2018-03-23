import pickle
import re

evaluate_num = 200
for index in range(10):
    index += 1
    task = pickle.load(open('./data/task_' + str(index), 'rb'))
    output = open('./data/task2_tieba_%d.txt' % int(index) , 'w')
    for i in range(evaluate_num):
        c = re.sub('<end>', '', task['tieba'][i][0])
        r = re.sub('<end>', '', task['tieba'][i][1])
        c = re.sub(' ', '', c)
        r = re.sub(' ', '', r)
        output.write(c.strip() + '[段落说明]\n')
        output.write(r.strip() + '[段落说明]\n')
        #output.write(r.strip() + '[量表题]\n' )
        output.write('%d/%d[量表题]\n' % (i+1, evaluate_num))
        output.write('1~5\n')
