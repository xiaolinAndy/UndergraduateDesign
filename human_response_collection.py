import pickle
import re

response_num = 20
index = input("请输入编号: ")
task1 = pickle.load(open('./data/task1_' + index, 'rb'))
output = open('./data/task1_res%d.txt' % int(index) , 'w')

print("请给出一个合理的对话回答（内容来自于twitter）")
for i in range(response_num):
    c = re.sub('</s>', '', task1['twitter'][i])
    print('(%d/%d) %s' % (i+1, response_num, c))
    res = input("请输入回答: ")
    print('\n')
    while res == '':
        res = input("请输入回答: ")
        print('\n')
    output.write(str(i+1) + '\n')
    output.write(res + '\n')

print("请给出一个合理的对话回答（内容来自于ubuntu）")
for i in range(response_num):
    c = re.sub('__eot__', '', task1['ubuntu'][i])
    c = re.sub('__eou__', '  ', c)
    print('(%d/%d) %s' % (i + 1, response_num, c))
    res = input("请输入回答: ")
    print('\n')
    while res == '':
        res = input("请输入回答: ")
        print('\n')
    output.write(str(i + 1) + '\n')
    output.write(res + '\n')

print("请给出一个合理的对话回答（内容来自于tieba）")
for i in range(response_num):
    c = re.sub('<end>', '', task1['tieba'][i])
    c = re.sub(' ', '', c)
    print('(%d/%d) %s' % (i + 1, response_num, c))
    res = input("请输入回答: ")
    print('\n')
    while res == '':
        res = input("请输入回答: ")
        print('\n')
    output.write(str(i + 1) + '\n')
    output.write(res + '\n')