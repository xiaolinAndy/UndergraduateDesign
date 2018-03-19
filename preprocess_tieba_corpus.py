import pickle
import re
import jieba

'''file_name = '../Data/comment_url_next.txt'
output = open('../Data/tieba_comments.pkl', 'wb')
f = open(file_name, 'r', encoding='utf-8')
dialogues = []
D = f.readlines()
context = []
max_len = 0
for i,line in enumerate(D):
    if i % 2 == 0:
        context = re.sub('^\d{n,}$','<num>',line).strip()
        max_len = max(max_len, len(context))
    else:
        response = re.sub('^\d{n,}$','<num>',line).strip()
        dialogue = [context, response]
        print(dialogue)
        dialogues.append(dialogue)
        max_len = max(max_len, len(response))
print(max_len)
print(len(dialogues))
pickle.dump(dialogues, output)'''
'''emoji_pattern = re.compile(
    u'(\ud83d[\ude00-\ude4f])|'  # emoticons
    u'(\ud83c[\udf00-\uffff])|'  # symbols & pictographs (1 of 2)
    u'(\ud83d[\u0000-\uddff])|'  # symbols & pictographs (2 of 2)
    u'(\ud83d[\ude80-\udeff])|'  # transport & map symbols
    u'(\ud83c[\udde0-\uddff])'  # flags (iOS)
    '+', flags=re.UNICODE)
file_name = '../Data/comment_url_next.txt'
output = open('../Data/tieba_comments_next.txt', 'w')
test = open('../Data/tieba_comments_test.txt', 'w')
dialogues = []
f = open(file_name, 'r', encoding='utf-8')
D = f.readlines()
count = 0
max_len = 0
tmp = 0
for i, line in enumerate(D):
    if line == '\n':
        continue
    if line != '#####\n':
        count += 1
    elif count == 2:
        #context = re.sub('^\d{n,}$','<num>',D[i-2]).strip()
        #response = re.sub('^\d{n,}$','<num>',D[i-1]).strip()
        #max_len = max(max_len, len(context), len(response))
        #dialogues.append([context, response])
        try:
            test.write(emoji_pattern.sub(r'', D[i-2]))
            test.write(emoji_pattern.sub(r'', D[i-1]))
        except:
            print(i, D[i-2], D[i-1])
            tmp += 1
            count = 0
            continue
        count = 0
        output.write(emoji_pattern.sub(r'', D[i-2]))
        output.write(emoji_pattern.sub(r'', D[i-1]))
    else:
        count = 0
print(tmp)
#print(len(dialogues))
#pickle.dump(dialogues, output)'''

D = open('../Data/tieba/comments_final.txt', 'r')
output = open('../Data/tieba/comments_final_modified.txt', 'w')
newf = open('../Data/tieba/comments_final_modified.pkl', 'wb')
D = D.readlines()
dialog = []
valid = 1
pair = []
for i, d in enumerate(D):
    utterance = d.strip()
    utterance = utterance + '<end>'
    utterance = re.sub('\d{5,}', '<num>', utterance)
    utterance = re.sub('&nbsp;', '', utterance)
    utterance = utterance.replace('& lt', '<')
    utterance = utterance.replace('& gt', '>')
    utterance = utterance.replace('&lt;', '<')
    utterance = utterance.replace('&gt;', '>')
    utterance = utterance.replace("`", " ")
    utterance = utterance.replace("..", ".")
    utterance = utterance.replace("..", ".")
    utterance = utterance.replace("..", ".")
    utterance = utterance.replace(",,", ",")
    utterance = utterance.replace(",,", ",")
    utterance = utterance.replace(",,", ",")
    utterance = utterance.replace('.', ' . ')
    utterance = utterance.replace('~', '')
    utterance = utterance.replace('*', '')
    utterance = utterance.replace('(', ' ')
    utterance = utterance.replace(')', ' ')
    utterance = utterance.replace('[', ' ')
    utterance = utterance.replace(']', ' ')
    utterance = re.sub('[\s]+', ' ', utterance)
    utterance = utterance.replace('  ', ' ')
    utterance = utterance.replace('  ', ' ')
    s = utterance
    while '! ! ! !' in s:
        s = s.replace('! ! ! !', '! ! !')
    while '。。。。' in s:
        s = s.replace('。。。。', '。。。')
    while '，，，，' in s:
        s = s.replace('，，，，', '，，，')
    while len(s) > 0 and s[-1] == ' ':
        s = s[0:-1]
    if not s[-5:] == '<end>':
        s = s + '<end>'
    try:
        output.write(s + '\n')
        cut = jieba.cut(s)
        l = []
        new = []
        index = 0
        for w in cut:
            l.append(w)
        while index < len(l):
            if index + 2 < len(l) and l[index] == '<' and l[index+1] == 'num' and l[index+2] == '>':
                new.append('<num>')
                index += 3
            elif index + 2 < len(l) and l[index] == '<' and l[index+1] == 'end' and l[index+2] == '>':
                new.append('<end>')
                index += 3
            elif l[index] != ' ':
                new.append(l[index])
                index += 1
            else:
                index += 1
        pair.append(new)
    except:
        print(s)
        valid = 0
    if i % 2 == 1 and valid:
        dialog.append(pair)
        pair = []
    elif i % 2 == 1:
        valid = 1
        pair = []
    if (i % 10000 == 0):
        print(i/10000)

pickle.dump(dialog, newf, protocol=2)

s = "我是林海涛，又名“海涛”，今天是：2018年5月10日+QQ<num>abc<end>。"
cut = jieba.cut(s)
l = []
new = []
index = 0
for w in cut:
    l.append(w)
while index < len(l):
    if index + 2 < len(l) and l[index] == '<' and l[index+1] == 'num' and l[index+2] == '>':
        new.append('<num>')
        index += 3
    elif index + 2 < len(l) and l[index] == '<' and l[index+1] == 'end' and l[index+2] == '>':
        new.append('<end>')
        index += 3
    else:
        new.append(l[index])
        index += 1
print(" | ".join(new))