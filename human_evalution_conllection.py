import pickle
import re

# 0 means unusable score
evaluate_num = 100
index = input("请输入编号: ")
task2 = pickle.load(open('./data/task2_' + index, 'rb'))
output = open('./data/task2_res%d.txt' % int(index) , 'a+')

for i in range(evaluate_num):
    c = re.sub('</s>', '', task2['twitter'][i][0])
    r = re.sub('</s>', '', task2['twitter'][i][1])
    print('(%d/%d) A: %s' % (i+1, evaluate_num, c))
    print('        B: %s' % (r))
    res = input("请评分: ")
    print('\n')
    while True:
        try:
            score = int(res)
            if score < 1 or score > 5:
                raise ValueError
            break
        except:
            res = input("输入有误，请重新评分: ")
            print('\n')
    output.write(str(i+1) + ': ' + res + '\n')