import pickle
import re


for index in range(10):
    task = pickle.load(open('./data/task_' + str(index + 1), 'rb'))
    evaluate_num = len(task['tieba'])
    output = open('./data/task_tieba_%d.txt' % int(index + 1) , 'w')
    for i in range(evaluate_num):
        c = re.sub('<end>', '', task['tieba'][i][0])
        r = re.sub('<end>', '', task['tieba'][i][1])
        c = re.sub(' ', '', c)
        r = re.sub(' ', '', r)
        c = re.sub('<num>', '[num]', c)
        r = re.sub('<num>', '[num]', r)
        output.write(c.strip() + '[段落说明]\n')
        output.write(r.strip() + '[段落说明]\n')
        #output.write(r.strip() + '[量表题]\n' )
        output.write('%d/%d[量表题]\n' % (i+1, evaluate_num))
        output.write('1~5\n')
