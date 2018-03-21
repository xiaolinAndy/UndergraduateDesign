import pickle
import re

response_num = 30
index = input("请输入编号: ")
task1 = pickle.load(open('./data/task1_' + index, 'rb'))
output1 = open('./data/task1_twitter_%d.txt' % int(index) , 'a+')
output2 = open('./data/task1_tieba_%d.txt' % int(index) , 'a+')

print("请给出一个合理的对话回答（内容来自于twitter）")
for i in range(response_num):
    c = re.sub('</s>', '', task1['twitter'][i])
    print('(%d/%d) %s' % (i+1, response_num, c))
    res = input("请输入回答: ")
    print('\n')
    output1.write(str(i+1) + ': ' + res + '\n')

print("请给出一个合理的对话回答（内容来自于tieba）")
for i in range(response_num):
    c = re.sub('<end>', '', task1['tieba'][i])
    c = re.sub(' ', '', c)
    print('(%d/%d) %s' % (i + 1, response_num, c))
    res = input("请输入回答: ")
    print('\n')
    output2.write(str(i + 1) + ': ' + res + '\n')