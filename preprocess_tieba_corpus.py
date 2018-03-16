import pickle
import re

'''file_name = '../Data/tieba_comments.txt'
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

'''file_name = '../Data/comment_url_1.txt'
output = open('../Data/tieba_comments_1.pkl', 'wb')
dialogues = []
f = open(file_name, 'r')
D = f.readlines()
count = 0
max_len = 0
for i, line in enumerate(D):
    if line == '\n':
        continue
    if line != '#####\n':
        count += 1
    elif count == 2:
        context = re.sub('^\d{n,}$','<num>',D[i-2]).strip()
        response = re.sub('^\d{n,}$','<num>',D[i-1]).strip()
        max_len = max(max_len, len(context), len(response))
        dialogues.append([context, response])
        count = 0
    else:
        count = 0
print(max_len)
print(len(dialogues))
pickle.dump(dialogues, output)'''

'''d1 = open('../Data/tieba_comments.pkl', 'rb')
d2 = open('../Data/tieba_comments_1.pkl', 'rb')
f = open('../Data/temp_comments_1.txt', 'w')
D1 = pickle.load(d1)
D2 = pickle.load(d2)
pickle.dump(D1+D2, open('../Data/tieba_comments_2.pkl', 'wb'), protocol=2)
for i,d in enumerate(D1+D2):
    try:
        f.write(d[0] + '\n')
        f.write(d[1] + '\n')
    except:
        continue'''

D = open('../Data/tieba_comments_2.pkl', 'rb')
output = open('../Data/tieba_comment_clean_txt', 'w')
new = open('../Data/tieba_comments_final.pkl', 'wb')
dialog = []
valid = 1
for i, d in enumerate(D):
    print(d.decode("gbk"))
    pair = []
    for j, s in enumerate(d):
        utterance = s + ' </s>'
        utterance = utterance.replace('& lt', '<')
        utterance = utterance.replace('& gt', '>')
        utterance = utterance.replace('&lt;', '<')
        utterance = utterance.replace('&gt;', '>')
        utterance = utterance.replace('"', ' " ')
        utterance = utterance.replace("'", " '")
        utterance = utterance.replace(";", " ")
        utterance = utterance.replace("`", " ")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace("..", ".")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace(",,", ",")
        utterance = utterance.replace('.', ' . ')
        utterance = utterance.replace('!', ' ! ')
        utterance = utterance.replace('?', ' ? ')
        utterance = utterance.replace(',', ' , ')
        utterance = utterance.replace('~', '')
        utterance = utterance.replace('-', ' - ')
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
        while len(s) > 0 and s[-1] == ' ':
            s = s[0:-1]
        if not s[-5:] == ' </s>':
            s = s + ' </s>'
        pair.append(s)
        try:
            output.write(s + '\n')
        except:
            print(s)
            valid = 0
    if valid:
        dialog.append(pair)
    else:
        valid = 1
    if (i % 10000 == 0):
        print(i/10000)

pickle.dump(dialog, new)